
CUDA_VISIBLE_DEVICES="1" python /root/autodl-tmp/zhengyuhan/EffectErase/examples/remove_wan/infer_remove_wan.py \
    --fg_bg_path /root/autodl-tmp/zhengyuhan/EffectErase/demo/FG_BG/video4.mp4 \
    --mask_path /root/autodl-tmp/zhengyuhan/EffectErase/demo/MASK/mask4.mp4 \
    --output_path /root/autodl-tmp/zhengyuhan/EffectErase/demo/REMOVE/remove4.mp4 \
    --text_encoder_path /root/autodl-tmp/zhengyuhan/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path /root/autodl-tmp/zhengyuhan/EffectErase/Model/Wan2.1_VAE.pth \
    --dit_path /root/autodl-tmp/zhengyuhan/EffectErase/Model/diffusion_pytorch_model.safetensors \
    --image_encoder_path /root/autodl-tmp/zhengyuhan/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path /root/autodl-tmp/zhengyuhan/EffectErase/EffectErase.ckpt