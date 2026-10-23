# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import logging
import os
import sys
import warnings
from datetime import datetime
import time
warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2.2"))
import wan2
from wan2.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan2.distributed.util import init_distributed_group
from wan2.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan2.utils.utils import merge_video_audio, save_video, str2bool

import imageio

import numpy as np
import cv2
import torchvision


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--high_lightx2v_path",
        type=str,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--low_lightx2v_path",
        type=str,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--high_lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--low_lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_lora_checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _cleanup_memory(tag: str = ""):
    """清理 Python 和 CUDA 缓存，避免循环生成时显存/内存累积导致卡死。"""
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if tag:
        try:
            rank = int(os.getenv("RANK", 0))
            if rank == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                logging.info(
                    f"[cleanup:{tag}] cuda allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
                )
        except Exception:
            pass


def render_video(tensor_fgr,
                tensor_pha,
                nrow=8,
                normalize=True,
                value_range=(-1, 1)):
    
    tensor_fgr = tensor_fgr.clamp(min(value_range), max(value_range))
    tensor_fgr = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor_fgr.unbind(2)
    ],
                            dim=1).permute(1, 2, 3, 0)
    tensor_fgr = (tensor_fgr * 255).type(torch.uint8).cpu()

    tensor_pha = tensor_pha.clamp(min(value_range), max(value_range))
    tensor_pha = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor_pha.unbind(2)
    ],
                            dim=1).permute(1, 2, 3, 0)
    tensor_pha = (tensor_pha * 255).type(torch.uint8).cpu()

    frames = []
    frames_fgr = []
    frames_pha = []
    for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
        frames_fgr.append(frame_fgr)
        frames_pha.append(frame_pha)

        frame_pha = (0.0 + frame_pha[:,:,0:1] + frame_pha[:,:,1:2] + frame_pha[:,:,2:3]) / 3.
        frame = np.concatenate([frame_fgr[:,:,::-1], frame_pha.astype(np.uint8)], axis=2)
        frames.append(frame)
    
    frames = frames[1:]
    frames_fgr = frames_fgr[1:]
    frames_pha = frames_pha[1:]

    # 生成灰白棋盘格图案
    def create_checkerboard(size=8, pattern_size=(512, 512)):
        # 创建棋盘格背景
        img = Image.new('RGB', (pattern_size[0], pattern_size[1]), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 填充棋盘格颜色
        for i in range(0, pattern_size[0], size):
            for j in range(0, pattern_size[1], size):
                if (i + j) // size % 2 == 0:
                    draw.rectangle([i, j, i+size, j+size], fill=(200, 200, 200))
        return img

    # 处理视频帧并替换透明部分为棋盘格
    def blender_background(frame_rgba, checkerboard):
        # 创建透明区域的掩码
        alpha_channel = frame_rgba[:, :, 3:] / 255. 
        checkerboard = np.array(checkerboard)
        checkerboard = cv2.resize(checkerboard, (frame_rgba.shape[1], frame_rgba.shape[0]))

        frame_rgb = frame_rgba[:, :, :3] * alpha_channel + checkerboard * (1-alpha_channel)
        return frame_rgb.astype(np.uint8)[:,:,::-1]
    
    checkerboard = create_checkerboard()
    white_bg = np.ones_like(frames[0][:,:,:3]) * 255
    balck_bg = np.zeros_like(frames[0][:,:,:3])

    video_checkerboard = [blender_background(f, checkerboard) for f in frames]
    video_white = [blender_background(f, white_bg) for f in frames]
    video_black = [blender_background(f, balck_bg) for f in frames]

    return video_checkerboard, video_white, video_black, frames_fgr, frames_pha, frames


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V_roll pipeline.")
        wan_t2v = wan2.WanT2V_roll(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            high_lightx2v=args.high_lightx2v_path,
            low_lightx2v=args.low_lightx2v_path,
            high_lora=args.high_lora_path,
            low_lora=args.low_lora_path,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        def save_video(save_path, target_frames, fps=16):
            writer = imageio.get_writer(
                save_path, fps=fps, codec='libx264', quality=8)
            for frame in target_frames:
                writer.append_data(frame)
            writer.close()


        logging.info(f"Generating video ...")
        with open(args.prompt_file) as fin:
            all_prompts = fin.read().splitlines()
        
        os.makedirs(args.output_dir, exist_ok=True)
        for args.prompt in all_prompts:
            logging.info(f"Input prompt: {args.prompt}")
            
            prompt_output_dir = os.path.join(args.output_dir, args.prompt)
            expected_file = os.path.join(prompt_output_dir, "fgr.mp4")
            skip_prompt = os.path.isdir(prompt_output_dir) and os.path.isfile(
                expected_file) and os.path.getsize(expected_file) > 0
            if dist.is_initialized():
                skip_prompt_list = [skip_prompt] if rank == 0 else [None]
                dist.broadcast_object_list(skip_prompt_list, src=0)
                skip_prompt = skip_prompt_list[0]
            if skip_prompt:
                if rank == 0:
                    logging.info(
                        f"Skip prompt '{args.prompt}': found complete outputs in {prompt_output_dir}"
                    )
                _cleanup_memory(tag=f"skip prompt: {args.prompt[:32]}")
                if dist.is_initialized():
                    dist.barrier()
                continue

            # 生成前先做一次清理，避免上一轮残留显存导致 OOM 或卡住
            _cleanup_memory(tag=f"before generate: {args.prompt[:32]}")

            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,               
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                output_dir=args.output_dir,)

            if rank == 0:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mov'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
                args.save_file = os.path.join(args.output_dir, args.save_file)


                logging.info(f"Saving generated video to {args.save_file}")

                video_fgr = video
                # 尽量把后处理放到 CPU 上进行，并立刻释放 GPU 上的拷贝
                try:
                    video_fgr = video_fgr.detach().to("cpu")
                except Exception:
                    pass
                frames_fgr = render_video(video_fgr[None], video_fgr[None])

                os.makedirs(f"{args.output_dir}/{args.prompt[:30]}", exist_ok=True)
                save_video(f"{args.output_dir}/{args.prompt[:30]}/fgr.mp4", frames_fgr)

                # 释放本轮用到的大张量引用
                del frames_fgr
                del video_fgr

            # 本轮生成结果也删掉，避免跨 iteration 引用残留
            del video
            _cleanup_memory(tag=f"after generate: {args.prompt[:32]}")

            # 多卡时保证同步，避免某张卡已完成但其他卡还在收尾，
            # 下一轮读取/释放时序错乱导致挂起
            if dist.is_initialized():
                dist.barrier()

        # 循环结束后再清理一次
        _cleanup_memory(tag="all prompts done")
        exit()
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        def save_video(save_path, target_frames, fps=16):
            writer = imageio.get_writer(
                save_path, fps=fps, codec='libx264', quality=8)
            for frame in target_frames:
                writer.append_data(frame)
            writer.close()
        wan_i2v = wan2.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            vae_lora_checkpoint=args.vae_lora_checkpoint,
            high_lightx2v=args.high_lightx2v_path,
            low_lightx2v=args.low_lightx2v_path,
            high_lora=args.high_lora_path,
            low_lora=args.low_lora_path,
            has_trans_encoder=True,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info("Generating video ...")
        img_list = [args.image]
        prompt_list = [args.prompt]
        for img, prompt in zip(img_list, prompt_list):
            args.prompt = prompt
            logging.info(f"Input prompt: {prompt}")
            start_time = time.time()

            random_seed = 0

            img_name = img.split("/")[-1].split(".")[0]
            img = Image.open(img).convert("RGBA")
            print("img size:", img.size)
            r, g, b, a = img.split()
            # 合并 RGB
            rgb_image = Image.merge('RGB', (r, g, b))

            # 将 Alpha 通道转换为 3 通道灰度图
            alpha_3ch = Image.merge('RGB', (a, a, a))

            # 创建一个白色背景图
            white_bg = Image.new("RGB", img.size, (255, 255, 255))

            # 将原图粘贴到白色背景上，使用 alpha 通道作为蒙版
            white_bg.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask

            pure_white = Image.new("RGB", img.size, (255, 255, 255))

            videos = wan_i2v.generate(
                args.prompt,
                white_bg,
                pure_white,
                white_bg,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                )
            
            if rank == 0:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mov'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
                args.save_file = os.path.join(args.output_dir, args.save_file)


                logging.info(f"Saving generated video to {args.save_file}")

                video_fgr, video_pha = videos
                # 尽量把后处理放到 CPU 上进行，并立刻释放 GPU 上的拷贝
                try:
                    video_fgr = video_fgr.detach().to("cpu")
                    video_pha = video_pha.detach().to("cpu")
                except Exception:
                    pass
                video_checkerboard, video_white, video_black, frames_fgr, frames_pha, frames =render_video(video_fgr[None], video_pha[None])

                os.makedirs(f"{args.output_dir}/{args.prompt[:50]}", exist_ok=True)
                save_video(f"{args.output_dir}/{args.prompt[:50]}/checkerboard.mp4", video_checkerboard)
                save_video(f"{args.output_dir}/{args.prompt[:50]}/pha.mp4", frames_pha)
                save_video(f"{args.output_dir}/{args.prompt[:50]}/fgr.mp4", frames_fgr)

                # 释放本轮用到的大张量引用
                del frames_fgr
                del video_fgr

            # 本轮生成结果也删掉，避免跨 iteration 引用残留
            _cleanup_memory(tag=f"after generate: {args.prompt[:32]}")

            # 多卡时保证同步，避免某张卡已完成但其他卡还在收尾，
            # 下一轮读取/释放时序错乱导致挂起
            if dist.is_initialized():
                dist.barrier()

        # 循环结束后再清理一次
        _cleanup_memory(tag="all prompts done")
        exit()

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    try:
        generate(args)
    finally:
        # 保险：无论正常退出或异常，都尽量释放显存
        try:
            _cleanup_memory(tag="exit")
        except Exception:
            pass
