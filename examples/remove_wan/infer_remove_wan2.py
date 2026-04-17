import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import imageio
import imageio.v3 as imageio_v3
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import rearrange

from diffsynth import ModelManager, WanRemovePipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch infer object/effect removal for folders containing video.mp4 and mask.mp4."
    )

    # I/O
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing sample folders. Each folder should contain video.mp4 and mask.mp4.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to save generated videos.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="removed.mp4",
        help="Output filename for each sample folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for sample folders under input_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    # Chunk / resolution settings
    parser.add_argument("--chunk_size", type=int, default=81, help="Model chunk length.")
    parser.add_argument(
        "--min_width",
        type=int,
        default=1280,
        help="Minimum output width after aspect-preserving upscale.",
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=720,
        help="Minimum output height after aspect-preserving upscale.",
    )
    parser.add_argument(
        "--size_multiple",
        type=int,
        default=16,
        help="Round output size to a multiple of this value.",
    )

    # Inference settings
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps.",
    )
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--use_teacache", action="store_true", help="Use TeaCache.")
    parser.add_argument(
        "--prompt_remove",
        type=str,
        default="Remove the specified object and all related effects, then restore a clean background.",
        help="Positive prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "blurry details, subtitles, watermark, painting, static frame, worst quality, low quality, JPEG artifacts, "
            "ugly, incomplete object removal, residual object, ghosting, flicker, distorted background, temporal inconsistency"
        ),
        help="Negative prompt.",
    )

    # Model paths
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to text encoder.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE.")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to DiT.")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="Path to image encoder.")
    parser.add_argument("--pretrained_lora_path", type=str, required=True, help="Path to pretrained LoRA.")

    return parser


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def get_target_size(
    orig_h: int,
    orig_w: int,
    min_h: int,
    min_w: int,
    size_multiple: int,
) -> Tuple[int, int]:
    scale = max(1.0, min_w / float(orig_w), min_h / float(orig_h))
    out_h = ceil_to_multiple(int(round(orig_h * scale)), size_multiple)
    out_w = ceil_to_multiple(int(round(orig_w * scale)), size_multiple)
    return out_h, out_w


def resize_pil(img: Image.Image, height: int, width: int, is_mask: bool = False) -> Image.Image:
    interpolation = (
        torchvision.transforms.InterpolationMode.NEAREST
        if is_mask
        else torchvision.transforms.InterpolationMode.BILINEAR
    )
    return torchvision.transforms.functional.resize(img, (height, width), interpolation=interpolation)


def frame_norm_to_tensor(frame: Image.Image) -> torch.Tensor:
    tensor = torchvision.transforms.functional.to_tensor(frame)
    tensor = torchvision.transforms.functional.normalize(
        tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    return tensor


def read_video_rgb_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames, fps


def rgb_array_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame)


def crop_square_from_pil(
    mask_img: Image.Image,
    fg_bg_img: Image.Image,
    target_size: int = 224,
    video_mask_path: Optional[str] = None,
) -> torch.Tensor:
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:
        mask_np = mask_np.max(axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8)

    img_np = np.array(fg_bg_img.convert("RGB"))
    h, w = mask_np.shape

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError(f"{video_mask_path or 'mask'} has no valid mask region")

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0

    side = max(bw, bh)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    sx0 = int(np.floor(cx - side / 2))
    sy0 = int(np.floor(cy - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    pad_left = max(0, -sx0)
    pad_top = max(0, -sy0)
    pad_right = max(0, sx1 - w)
    pad_bottom = max(0, sy1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        mask_np = np.pad(
            mask_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        sx0 += pad_left
        sx1 += pad_left
        sy0 += pad_top
        sy1 += pad_top

    crop_img = img_np[sy0:sy1, sx0:sx1]
    crop_mask = mask_np[sy0:sy1, sx0:sx1][..., None]
    crop_img = crop_img * crop_mask

    crop_t = torch.from_numpy(crop_img).permute(2, 0, 1).float().unsqueeze(0)
    crop_t = F.interpolate(
        crop_t,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    crop_t = crop_t / 255.0
    crop_t = crop_t * 2.0 - 1.0
    return crop_t


def has_mask(mask_frame: np.ndarray) -> bool:
    if mask_frame.ndim == 3:
        return bool(mask_frame.max() > 0)
    return bool(np.max(mask_frame) > 0)


def get_reference_crop_from_chunk(
    mask_chunk: Sequence[np.ndarray],
    video_chunk: Sequence[np.ndarray],
    fallback_crop: Optional[torch.Tensor],
    mask_path_for_log: str,
) -> torch.Tensor:
    for mask_frame, video_frame in zip(mask_chunk, video_chunk):
        if has_mask(mask_frame):
            return crop_square_from_pil(
                rgb_array_to_pil(mask_frame),
                rgb_array_to_pil(video_frame),
                target_size=224,
                video_mask_path=mask_path_for_log,
            )
    if fallback_crop is None:
        raise ValueError(f"No valid mask region found in chunk and no fallback crop: {mask_path_for_log}")
    return fallback_crop


def prepare_chunk_tensor(
    frames: Sequence[np.ndarray],
    target_h: int,
    target_w: int,
    chunk_size: int,
    is_mask: bool,
) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("Empty chunk.")

    padded_frames = list(frames)
    while len(padded_frames) < chunk_size:
        padded_frames.append(padded_frames[-1].copy())

    tensor_list = []
    for frame in padded_frames:
        pil = rgb_array_to_pil(frame)
        pil = resize_pil(pil, target_h, target_w, is_mask=is_mask)
        tensor_list.append(frame_norm_to_tensor(pil))

    frames_tensor = torch.stack(tensor_list, dim=0)
    frames_tensor = rearrange(frames_tensor, "T C H W -> C T H W")
    return frames_tensor


def save_frames_as_video_with_ref_fps(
    frames: Sequence[np.ndarray],
    output_path: str,
    ref_video_path: Optional[str] = None,
    default_fps: float = 24.0,
):
    if ref_video_path:
        try:
            reader = imageio.get_reader(ref_video_path)
            meta = reader.get_meta_data()
            fps = meta.get("fps", default_fps)
            reader.close()
        except Exception as e:
            print(f"[WARN] Failed to read fps from ref video: {e}, use default_fps={default_fps}")
            fps = default_fps
    else:
        fps = default_fps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output_path.as_posix(),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        quality=8,
    )

    try:
        for frame in frames:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"[INFO] Saved video to: {output_path} (fps={fps})")
    return output_path.as_posix()


def list_sample_dirs(input_root: str, recursive: bool) -> List[Path]:
    input_root = Path(input_root)
    if recursive:
        video_files = sorted(input_root.rglob("video.mp4"))
    else:
        video_files = sorted(p / "video.mp4" for p in input_root.iterdir() if p.is_dir())

    sample_dirs = []
    for video_file in video_files:
        if video_file.is_file() and (video_file.parent / "mask.mp4").is_file():
            sample_dirs.append(video_file.parent)
    return sample_dirs


def build_output_path(sample_dir: Path, input_root: str, output_root: str, output_name: str) -> Path:
    rel_dir = sample_dir.relative_to(Path(input_root))
    return Path(output_root) / rel_dir / output_name


def run_single_sample(
    pipe: WanRemovePipeline,
    args,
    sample_dir: Path,
    device: torch.device,
) -> None:
    video_path = str(sample_dir / "video.mp4")
    mask_path = str(sample_dir / "mask.mp4")
    output_path = build_output_path(sample_dir, args.input_root, args.output_root, args.output_name)

    if output_path.exists() and not args.overwrite:
        print(f"[SKIP] Output exists: {output_path}")
        return

    print(f"\n[INFO] Processing sample: {sample_dir}")
    start_t = time.time()

    video_frames, fps = read_video_rgb_frames(video_path)
    mask_frames, mask_fps = read_video_rgb_frames(mask_path)

    if len(video_frames) != len(mask_frames):
        raise ValueError(
            f"Frame count mismatch between video and mask: {video_path} ({len(video_frames)}) vs {mask_path} ({len(mask_frames)})"
        )

    orig_h, orig_w = video_frames[0].shape[:2]
    target_h, target_w = get_target_size(
        orig_h=orig_h,
        orig_w=orig_w,
        min_h=args.min_height,
        min_w=args.min_width,
        size_multiple=args.size_multiple,
    )

    print(
        f"[INFO] video frames={len(video_frames)}, fps={fps:.4f}, original={orig_w}x{orig_h}, target={target_w}x{target_h}"
    )
    if abs(fps - mask_fps) > 1e-3:
        print(f"[WARN] video fps ({fps:.4f}) != mask fps ({mask_fps:.4f}); output will follow source video fps.")

    total_frames = len(video_frames)
    chunk_size = args.chunk_size
    out_frames: List[np.ndarray] = []

    global_ref_crop = get_reference_crop_from_chunk(
        mask_chunk=mask_frames,
        video_chunk=video_frames,
        fallback_crop=None,
        mask_path_for_log=mask_path,
    )

    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        video_chunk = video_frames[start:end]
        mask_chunk = mask_frames[start:end]
        valid_len = len(video_chunk)

        ref_crop = get_reference_crop_from_chunk(
            mask_chunk=mask_chunk,
            video_chunk=video_chunk,
            fallback_crop=global_ref_crop,
            mask_path_for_log=mask_path,
        )

        video_tensor = prepare_chunk_tensor(
            frames=video_chunk,
            target_h=target_h,
            target_w=target_w,
            chunk_size=chunk_size,
            is_mask=False,
        ).to(device)
        mask_tensor = prepare_chunk_tensor(
            frames=mask_chunk,
            target_h=target_h,
            target_w=target_w,
            chunk_size=chunk_size,
            is_mask=True,
        ).to(device)
        ref_crop = ref_crop.to(device)

        print(f"[INFO]  chunk {start:05d}:{end:05d} (valid={valid_len}, padded_to={chunk_size})")
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            remove_video, _ = pipe(
                video_mask=mask_tensor,
                video_fg_bg=video_tensor,
                video_bg=None,
                task="remove",
                fg_first_img=ref_crop,
                prompt_remove=args.prompt_remove,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg,
                seed=args.seed,
                tiled=args.tiled,
                height=target_h,
                width=target_w,
                tea_cache_l1_thresh=0.3 if args.use_teacache else None,
                tea_cache_model_id="Wan2.1-T2V-1.3B" if args.use_teacache else None,
            )

        remove_video = np.asarray(remove_video)
        if remove_video.ndim != 4:
            raise ValueError(f"Unexpected model output shape: {remove_video.shape}")
        if len(remove_video) < valid_len:
            raise ValueError(
                f"Model output shorter than valid chunk length: got {len(remove_video)}, need {valid_len}"
            )

        out_frames.extend(remove_video[:valid_len])

        del video_tensor, mask_tensor, ref_crop, remove_video
        torch.cuda.empty_cache()

    if len(out_frames) != total_frames:
        raise RuntimeError(
            f"Output frame count mismatch for {sample_dir}: output={len(out_frames)} vs input={total_frames}"
        )

    save_frames_as_video_with_ref_fps(
        out_frames,
        output_path=str(output_path),
        ref_video_path=video_path,
        default_fps=fps,
    )

    print(
        f"[INFO] Done sample: {sample_dir.name} | frames={total_frames} | cost={time.time() - start_t:.2f}s"
    )


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently requires CUDA.")

    sample_dirs = list_sample_dirs(args.input_root, args.recursive)
    if not sample_dirs:
        raise RuntimeError(
            f"No sample folders found under {args.input_root}. Each sample must contain video.mp4 and mask.mp4."
        )

    print(f"[INFO] Found {len(sample_dirs)} sample folders.")
    print("[INFO] Building model...")
    model_manager = ModelManager(device="cuda")
    model_manager.load_models(
        [
            args.dit_path,
            args.text_encoder_path,
            args.vae_path,
            args.image_encoder_path,
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora_v2(args.pretrained_lora_path, lora_alpha=args.lora_alpha)

    pipe = WanRemovePipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

    num_ok = 0
    num_fail = 0
    failed_samples = []

    for sample_dir in sample_dirs:
        try:
            run_single_sample(pipe=pipe, args=args, sample_dir=sample_dir, device=device)
            num_ok += 1
        except Exception as e:
            num_fail += 1
            failed_samples.append((str(sample_dir), str(e)))
            print(f"[ERROR] Failed sample: {sample_dir}")
            print(f"[ERROR] {e}")
            torch.cuda.empty_cache()

    print("\n========== SUMMARY ==========")
    print(f"[INFO] Success: {num_ok}")
    print(f"[INFO] Failed : {num_fail}")
    if failed_samples:
        print("[INFO] Failed samples:")
        for path, err in failed_samples:
            print(f"  - {path}: {err}")


if __name__ == "__main__":
    main()
