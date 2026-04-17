import argparse
import gc
import inspect
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Reduce CUDA allocator fragmentation before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import imageio
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
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory containing sample folders. Each folder should contain video.mp4 and mask.mp4.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory to save generated videos.")
    parser.add_argument("--output_name", type=str, default="removed.mp4",
                        help="Output filename for each sample folder.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search for sample folders under input_root.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs.")
    parser.add_argument("--keep_partial_on_error", action="store_true",
                        help="Keep partially written temp video if a later chunk fails.")

    # Resolution settings
    parser.add_argument("--target_width", type=int, default=1280,
                        help="Reference width. Large videos are downscaled to fit within this size; small videos are upscaled to reach at least this size or the paired target_height, while preserving aspect ratio.")
    parser.add_argument("--target_height", type=int, default=720,
                        help="Reference height. Large videos are downscaled to fit within this size; small videos are upscaled to reach at least this size or the paired target_width, while preserving aspect ratio.")
    parser.add_argument("--size_multiple", type=int, default=16,
                        help="Round output size to a multiple of this value.")

    # Chunk / inference settings
    parser.add_argument("--chunk_size", type=int, default=21,
                        help="Initial chunk length. Smaller values use less VRAM.")
    parser.add_argument("--min_chunk_size", type=int, default=9,
                        help="Smallest chunk length allowed during automatic OOM fallback.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling from the start.")
    parser.add_argument("--auto_enable_tiled_on_oom", action="store_true",
                        help="If an OOM happens and --tiled was not set, retry the same chunk with tiled=True.")
    parser.add_argument("--use_teacache", action="store_true", help="Use TeaCache.")
    parser.add_argument("--prompt_remove", type=str,
                        default="Remove the specified object and all related effects, then restore a clean background.",
                        help="Positive prompt.")
    parser.add_argument("--negative_prompt", type=str,
                        default=(
                            "blurry details, subtitles, watermark, painting, static frame, worst quality, low quality, JPEG artifacts, "
                            "ugly, incomplete object removal, residual object, ghosting, flicker, distorted background, temporal inconsistency"
                        ),
                        help="Negative prompt.")
    parser.add_argument("--persistent_param_in_dit", type=int, default=2 * 10**9,
                        help="Budget passed to enable_vram_management. Lower values usually save more VRAM.")

    # Model paths
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to text encoder.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE.")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to DiT.")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="Path to image encoder.")
    parser.add_argument("--pretrained_lora_path", type=str, required=True, help="Path to pretrained LoRA.")

    return parser


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def floor_to_multiple(x: int, m: int) -> int:
    return max(m, int(math.floor(x / m) * m))


# Many video diffusion pipelines are happiest with lengths of the form 4k+1.
def normalize_chunk_size(n: int) -> int:
    n = max(1, int(n))
    if n == 1:
        return 1
    return ((n - 1) // 4) * 4 + 1


def next_smaller_chunk_size(current: int, min_chunk_size: int) -> int:
    min_chunk_size = normalize_chunk_size(min_chunk_size)
    if current <= min_chunk_size:
        return current

    halved = normalize_chunk_size(max(min_chunk_size, current // 2))
    if halved < current:
        return halved

    stepped = normalize_chunk_size(max(min_chunk_size, current - 4))
    return stepped if stepped < current else current


def get_target_size(orig_h: int, orig_w: int, ref_h: int, ref_w: int, size_multiple: int) -> Tuple[int, int]:
    # If the video is larger than the reference box, downscale to fit in it.
    # If smaller, upscale until at least one reference side is reached.
    if orig_w > ref_w or orig_h > ref_h:
        scale = min(ref_w / float(orig_w), ref_h / float(orig_h))
        out_h = floor_to_multiple(int(round(orig_h * scale)), size_multiple)
        out_w = floor_to_multiple(int(round(orig_w * scale)), size_multiple)
    elif orig_w < ref_w or orig_h < ref_h:
        scale = max(ref_w / float(orig_w), ref_h / float(orig_h))
        out_h = ceil_to_multiple(int(round(orig_h * scale)), size_multiple)
        out_w = ceil_to_multiple(int(round(orig_w * scale)), size_multiple)
    else:
        out_h = ceil_to_multiple(orig_h, size_multiple)
        out_w = ceil_to_multiple(orig_w, size_multiple)

    return max(size_multiple, out_h), max(size_multiple, out_w)


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


def crop_square_from_pil(mask_img: Image.Image, fg_bg_img: Image.Image, target_size: int = 224,
                         video_mask_path: Optional[str] = None) -> torch.Tensor:
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
    crop_t = F.interpolate(crop_t, size=(target_size, target_size), mode="bilinear", align_corners=False)[0]
    crop_t = crop_t / 255.0
    crop_t = crop_t * 2.0 - 1.0
    return crop_t


def has_mask(mask_frame: np.ndarray) -> bool:
    if mask_frame.ndim == 3:
        return bool(mask_frame.max() > 0)
    return bool(np.max(mask_frame) > 0)


def get_reference_crop_from_chunk(mask_chunk: Sequence[np.ndarray], video_chunk: Sequence[np.ndarray],
                                  fallback_crop: Optional[torch.Tensor], mask_path_for_log: str) -> torch.Tensor:
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


def prepare_chunk_tensor(frames: Sequence[np.ndarray], target_h: int, target_w: int,
                         chunk_size: int, is_mask: bool) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("Empty chunk.")

    padded_frames = list(frames)
    while len(padded_frames) < chunk_size:
        padded_frames.append(padded_frames[-1])

    tensors = []
    for frame in padded_frames:
        pil = rgb_array_to_pil(frame)
        pil = resize_pil(pil, target_h, target_w, is_mask=is_mask)
        tensors.append(frame_norm_to_tensor(pil))

    stacked = torch.stack(tensors, dim=0)      # [T,C,H,W]
    stacked = rearrange(stacked, "T C H W -> C T H W")
    return stacked


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


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in text


def detect_num_frames_kwarg(pipe: WanRemovePipeline) -> Optional[str]:
    try:
        sig = inspect.signature(pipe.__call__)
        for name in ("num_frames", "video_num_frames", "num_video_frames"):
            if name in sig.parameters:
                return name
    except Exception:
        pass
    return None


def infer_one_chunk(pipe: WanRemovePipeline, args, device: torch.device,
                    video_chunk: Sequence[np.ndarray], mask_chunk: Sequence[np.ndarray], ref_crop: torch.Tensor,
                    target_h: int, target_w: int, chunk_size: int, tiled: bool,
                    num_frames_kwarg: Optional[str]) -> np.ndarray:
    video_tensor = prepare_chunk_tensor(video_chunk, target_h, target_w, chunk_size, is_mask=False).to(device)
    mask_tensor = prepare_chunk_tensor(mask_chunk, target_h, target_w, chunk_size, is_mask=True).to(device)
    ref_crop = ref_crop.to(device)

    extra_kwargs = {}
    if num_frames_kwarg is not None:
        extra_kwargs[num_frames_kwarg] = chunk_size

    try:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
                tiled=tiled,
                height=target_h,
                width=target_w,
                tea_cache_l1_thresh=0.3 if args.use_teacache else None,
                tea_cache_model_id="Wan2.1-T2V-1.3B" if args.use_teacache else None,
                **extra_kwargs,
            )
        return np.asarray(remove_video)
    finally:
        del video_tensor, mask_tensor, ref_crop
        cleanup_cuda()


def open_video_writer(output_path: Path, fps: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # imageio-ffmpeg infers the container from the filename extension.
    # A path ending with ".mp4.part" may be rejected because the final suffix is ".part".
    # Keep the real video suffix at the end, e.g. "removed.part.mp4".
    temp_path = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
    if temp_path.exists():
        temp_path.unlink()
    writer = imageio.get_writer(
        temp_path.as_posix(),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        quality=8,
    )
    return writer, temp_path


def finalize_video_writer(writer, temp_path: Path, final_path: Path):
    writer.close()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    temp_path.replace(final_path)


def abort_video_writer(writer, temp_path: Path, keep_partial: bool):
    try:
        writer.close()
    except Exception:
        pass
    if not keep_partial and temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass


def run_single_sample(pipe: WanRemovePipeline, args, sample_dir: Path, device: torch.device,
                      num_frames_kwarg: Optional[str]) -> None:
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
        ref_h=args.target_height,
        ref_w=args.target_width,
        size_multiple=args.size_multiple,
    )

    print(f"[INFO] video frames={len(video_frames)}, fps={fps:.4f}, original={orig_w}x{orig_h}, target={target_w}x{target_h}")
    if abs(fps - mask_fps) > 1e-3:
        print(f"[WARN] video fps ({fps:.4f}) != mask fps ({mask_fps:.4f}); output will follow source video fps.")

    total_frames = len(video_frames)
    runtime_chunk_size = normalize_chunk_size(args.chunk_size)
    min_chunk_size = normalize_chunk_size(args.min_chunk_size)
    if runtime_chunk_size < min_chunk_size:
        runtime_chunk_size = min_chunk_size

    global_ref_crop = get_reference_crop_from_chunk(
        mask_chunk=mask_frames,
        video_chunk=video_frames,
        fallback_crop=None,
        mask_path_for_log=mask_path,
    )

    writer, temp_path = open_video_writer(output_path, fps)
    num_written = 0
    start = 0
    current_tiled = bool(args.tiled)

    try:
        while start < total_frames:
            end = min(start + runtime_chunk_size, total_frames)
            video_chunk = video_frames[start:end]
            mask_chunk = mask_frames[start:end]
            valid_len = len(video_chunk)

            ref_crop = get_reference_crop_from_chunk(
                mask_chunk=mask_chunk,
                video_chunk=video_chunk,
                fallback_crop=global_ref_crop,
                mask_path_for_log=mask_path,
            )

            print(f"[INFO]  chunk {start:05d}:{end:05d} (valid={valid_len}, padded_to={runtime_chunk_size}, tiled={current_tiled})")

            try:
                remove_video = infer_one_chunk(
                    pipe=pipe,
                    args=args,
                    device=device,
                    video_chunk=video_chunk,
                    mask_chunk=mask_chunk,
                    ref_crop=ref_crop,
                    target_h=target_h,
                    target_w=target_w,
                    chunk_size=runtime_chunk_size,
                    tiled=current_tiled,
                    num_frames_kwarg=num_frames_kwarg,
                )
            except Exception as e:
                if not is_oom_error(e):
                    raise

                print(f"[WARN] OOM at chunk_size={runtime_chunk_size}, tiled={current_tiled}: {e}")
                cleanup_cuda()

                if (not current_tiled) and args.auto_enable_tiled_on_oom:
                    current_tiled = True
                    print("[WARN] Retrying same chunk with tiled=True.")
                    continue

                smaller = next_smaller_chunk_size(runtime_chunk_size, min_chunk_size)
                if smaller >= runtime_chunk_size:
                    raise RuntimeError(
                        f"OOM persists even at smallest allowed chunk_size={runtime_chunk_size}. "
                        f"Try lowering --num_inference_steps, enabling --tiled, or reducing target resolution."
                    ) from e

                runtime_chunk_size = smaller
                print(f"[WARN] Retrying same chunk with smaller chunk_size={runtime_chunk_size}.")
                continue

            if remove_video.ndim != 4:
                raise ValueError(f"Unexpected model output shape: {remove_video.shape}")
            if len(remove_video) < valid_len:
                raise ValueError(f"Model output shorter than valid chunk length: got {len(remove_video)}, need {valid_len}")

            for frame in remove_video[:valid_len]:
                writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
            num_written += valid_len

            del remove_video
            cleanup_cuda()
            start = end

        if num_written != total_frames:
            raise RuntimeError(f"Output frame count mismatch for {sample_dir}: output={num_written} vs input={total_frames}")

        finalize_video_writer(writer, temp_path, output_path)
        print(f"[INFO] Saved video to: {output_path} (fps={fps})")
        print(f"[INFO] Done sample: {sample_dir.name} | frames={total_frames} | final_chunk_size={runtime_chunk_size} | cost={time.time() - start_t:.2f}s")
    except Exception:
        abort_video_writer(writer, temp_path, keep_partial=args.keep_partial_on_error)
        raise


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
        [args.dit_path, args.text_encoder_path, args.vae_path, args.image_encoder_path],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora_v2(args.pretrained_lora_path, lora_alpha=args.lora_alpha)

    pipe = WanRemovePipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=args.persistent_param_in_dit)
    num_frames_kwarg = detect_num_frames_kwarg(pipe)
    if num_frames_kwarg is not None:
        print(f"[INFO] Detected pipeline frame-length argument: {num_frames_kwarg}")
    else:
        print("[WARN] No explicit frame-length argument detected in pipeline signature. If chunk_size != model default, a shape mismatch may still occur.")

    num_ok = 0
    num_fail = 0
    failed_samples = []

    for sample_dir in sample_dirs:
        try:
            run_single_sample(pipe=pipe, args=args, sample_dir=sample_dir, device=device, num_frames_kwarg=num_frames_kwarg)
            num_ok += 1
        except Exception as e:
            num_fail += 1
            failed_samples.append((str(sample_dir), str(e)))
            print(f"[ERROR] Failed sample: {sample_dir}")
            print(f"[ERROR] {e}")
            cleanup_cuda()

    print("\n========== SUMMARY ==========")
    print(f"[INFO] Success: {num_ok}")
    print(f"[INFO] Failed : {num_fail}")
    if failed_samples:
        print("[INFO] Failed samples:")
        for path, err in failed_samples:
            print(f"  - {path}: {err}")


if __name__ == "__main__":
    main()
