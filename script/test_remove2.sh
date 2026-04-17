# CUDA_VISIBLE_DEVICES="1" python /data/zyh/EffectErase/examples/remove_wan/infer_remove_wan2.py \
#     --input_root /data/zyh/EffectErase/Input/videos \
#     --output_root /data/zyh/EffectErase/Output/videos \
#     --recursive \
#     --text_encoder_path /data/zyh/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
#     --vae_path /data/zyh/EffectErase/Model/Wan2.1_VAE.pth \
#     --dit_path /data/zyh/EffectErase/Model/diffusion_pytorch_model.safetensors \
#     --image_encoder_path /data/zyh/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#     --pretrained_lora_path /data/zyh/EffectErase/EffectErase.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/zyh/EffectErase/examples/remove_wan/infer_remove_wan_multigpu.py \
    --input_root /data/zyh/EffectErase/Input/videos \
    --output_root /data/zyh/EffectErase/Output/videos \
    --recursive \
    --gpu_ids 0,1,2,3 \
    --chunk_size 81 \
    --target_width 1280 \
    --target_height 720 \
    --tiled \
    --text_encoder_path /data/zyh/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path /data/zyh/EffectErase/Model/Wan2.1_VAE.pth \
    --dit_path /data/zyh/EffectErase/Model/diffusion_pytorch_model.safetensors \
    --image_encoder_path /data/zyh/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path /data/zyh/EffectErase/EffectErase.ckpt
   
# CUDA_VISIBLE_DEVICES="1" python /data/zyh/EffectErase/examples/remove_wan/infer_remove_wan.py \
#     --fg_bg_path /data/zyh/EffectErase/demo/FG_BG/video4.mp4 \
#     --mask_path /data/zyh/EffectErase/demo/MASK/mask4.mp4 \
#     --output_path /data/zyh/EffectErase/demo/REMOVE/remove4.mp4 \
#     --text_encoder_path /data/zyh/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
#     --vae_path /data/zyh/EffectErase/Model/Wan2.1_VAE.pth \
#     --dit_path /data/zyh/EffectErase/Model/diffusion_pytorch_model.safetensors \
#     --image_encoder_path /data/zyh/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#     --pretrained_lora_path /data/zyh/EffectErase/EffectErase.ckpt

# CUDA_VISIBLE_DEVICES="1" python /data/zyh/EffectErase/examples/remove_wan/infer_remove_wan3.py \
#     --input_root /data/zyh/EffectErase/Input/videos \
#     --output_root /data/zyh/EffectErase/Output/videos \
#     --recursive \
#     --chunk_size 21 \
#     --min_chunk_size 9 \
#     --num_inference_steps 30 \
#     --tiled \
#     --auto_enable_tiled_on_oom \
#     --text_encoder_path /data/zyh/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
#     --vae_path /data/zyh/EffectErase/Model/Wan2.1_VAE.pth \
#     --dit_path /data/zyh/EffectErase/Model/diffusion_pytorch_model.safetensors \
#     --image_encoder_path /data/zyh/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#     --pretrained_lora_path /data/zyh/EffectErase/EffectErase.ckpt

CUDA_VISIBLE_DEVICES="1" python /data/zyh/EffectErase/examples/remove_wan/infer_remove_wan3.py \
    --input_root /data/zyh/EffectErase/Input/videos \
    --output_root /data/zyh/EffectErase/Output/videos \
    --recursive \
    --chunk_size 21 \
    --min_chunk_size 9 \
    --target_width 1280 \
    --target_height 720 \
    --tiled \
    --auto_enable_tiled_on_oom \
    --keep_partial_on_error \
    --text_encoder_path /data/zyh/EffectErase/Model/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path /data/zyh/EffectErase/Model/Wan2.1_VAE.pth \
    --dit_path /data/zyh/EffectErase/Model/diffusion_pytorch_model.safetensors \
    --image_encoder_path /data/zyh/EffectErase/Model/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --pretrained_lora_path /data/zyh/EffectErase/EffectErase.ckpt