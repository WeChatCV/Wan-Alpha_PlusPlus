if ! python3 -c "pip install easydict" &> /dev/null; then
    pip install xfuser==0.4.1 --no-deps
    pip install easydict
    pip install dashscope
    pip install diffusers==0.31.0
    pip install flash_attn

    pip install -r requirements.txt
else
    echo "pip requirements installed."
fi


output_path="checkpoints"
mkdir -p $output_path

torchrun --nproc_per_node=8  generate_2.2_new.py --task i2v-A14B --size 1280*720 \
        --ckpt_dir Wan-AI/Wan2.2-I2V-A14B \
        --dit_fsdp --t5_fsdp --ulysses_size 8 \
        --frame_num 49 \
        --vae_lora_checkpoint "VAE/pytorch_model.bin" \
        --high_lora high_noise.safetensors \
        --low_lora low_noise.safetensors \
        --output_dir $output_path/result  2>&1 | tee -a $output_path/result.log

