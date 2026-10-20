<div align="center">

  <h1>
    Wan-Alpha++
  </h1>

  <h3>Animating a Single RGBA Asset into Controllable Transparent Video Effects</h3>



[![arXiv](https://img.shields.io/badge/arXiv-2610.26189-red)](https://arxiv.org/abs/2610.26189)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/htdong/Wan-Alpha-I2V)



</div>


>Extensive experiments demonstrate that Wan-Alpha++ achieves state-of-the-art performance in image-to-RGBA generation, precise spatial transparency control, and inference efficiency.

---

### 🔥 News
* **[2026.10.19]** We release the paper of Wan-Alpha++ on [arXiv](https://arxiv.org/abs/2610.26189).
* **[2026.10.19]** Released checkpoints for normal RGBA video of Wan-Alpha++ on [GitHub](https://github.com/WeChatCV/Wan-Alpha++).


---

### 📝 To-Do List

- [x] **Paper**: Available on [arXiv](https://arxiv.org/abs/2610.26189).
- [x] **Inference Code**: Released inference pipeline for normal RGBA video of Wan-Alpha++.
- [x] **Model Weights of Normal RGBA Video**: Released checkpoints for Wan-Alpha++.
- [] Training Code.
- [] Checkpoints for Effect RGBA Video.




### 🚀 Quick Start

##### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/WeChatCV/Wan-Alpha++.git
cd Wan-Alpha++

# Create and activate Conda environment
conda create -n Wan-Alpha++ python=3.11 -y
conda activate Wan-Alpha++

# Install dependencies
pip install -r requirements.txt
```

##### 2. Model Download
Download [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)


Download [Wan-Alpha++](https://huggingface.co/htdong/Wan-Alpha-I2V)

Download [Wan-Alpha-v2.0-VAE](https://huggingface.co/htdong/Wan-Alpha-VAE/blob/main/Wan-Alpha-v2-VAE.bin)

### 🧪 Usage
You can test our model through:
```bash
torchrun --nproc_per_node=8  generate_2.2_new.py \
        --task i2v-A14B --size 832*480 \
        --ckpt_dir Wan-AI/Wan2.2-I2V-A14B \
        --dit_fsdp --t5_fsdp --ulysses_size 8 \
        --frame_num 49 \
        --vae_lora_checkpoint "VAE/pytorch_model.bin" \
        --high_lora high_noise.safetensors \
        --low_lora low_noise.safetensors \
        --output_dir $output_path/result
```
You can specify the weights of `Wan2.2-I2V-A14B` with `--ckpt_dir`, `Wan-Alpha-VAE` with `--vae_lora_checkpoint`, and `Wan-Alpha-I2V` with `--high_lora` and `--low_lora`. Finally, you can find the rendered RGBA videos with a checkerboard background and PNG frames at `--output_dir`.


**Prompt Writing Tip:**  You need to specify that the background of the video is transparent, the visual style, the shot type (such as close-up, medium shot, wide shot, or extreme close-up), and a description of the main subject. Prompts support both Chinese and English input.

```bash
# An example of prompt.
This video has a transparent background. A woman holds a clear cup containing orange juice; she lifts the cup, takes a sip, and smiles.
```




### 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
* [Wan2.2](https://github.com/Wan-Video/Wan2.2) (base video generation model)


We sincerely thank the authors and contributors of these projects.


### ✏ Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@misc{dong2026wanalphaanimatingsinglergba,
      title={Wan-Alpha++: Animating a Single RGBA Asset into Controllable Transparent Video Effects}, 
      author={Haotian Dong and Xin Wang and Wenjing Wang and Chen Li and Jing Lyu and Yan Tina Luximon and Ping Li and Wanchao Su and Ke Xu and Di Lin},
      year={2026},
      eprint={2610.26189},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.26189}, 
}
``` 

### 📬 Contact Us

If you have any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/WeChatCV/Wan-Alpha++/issues) . We look forward to your feedback!
