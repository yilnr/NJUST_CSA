import argparse
import inspect
import math
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
        description=(
            "Batch infer object/effect removal for folders containing video.mp4 and mask.mp4. "
            "Supports chunked inference and sample-level multi-GPU parallelism."
        )
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

    # Multi-GPU
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help='Comma-separated GPU ids for sample-level parallel processing, e.g. "0,1,2,3".',
    )

    # Chunk / resolution settings
    parser.add_argument("--chunk_size", type=int, default=21, help="Frames per inference chunk.")
    parser.add_argument(
        "--target_width",
        type=int,
        default=1280,
        help="Maximum inference width for large videos; smaller videos can optionally be upscaled to this width.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=720,
        help="Maximum inference height for large videos; smaller videos can optionally be upscaled to this height.",
    )
    parser.add_argument(
        "--size_multiple",
        type=int,
        default=16,
        help="Round inference size to a multiple of this value.",
    )
    parser.add_argument(
        "--upscale_small_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to upscale videos smaller than target_width/target_height. Default: True.",
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


def round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))


def clamp_round_to_multiple(x: int, m: int, max_x: Optional[int] = None) -> int:
    out = round_to_multiple(x, m)
    if max_x is not None and out > max_x:
        out = max(m, int(math.floor(max_x / m) * m))
    return out


def get_infer_size(
    orig_h: int,
    orig_w: int,
    target_h: int,
    target_w: int,
    size_multiple: int,
    upscale_small_videos: bool,
) -> Tuple[int, int]:
    """
    Preserve aspect ratio.

    - Large videos are scaled down to fit within target_h x target_w.
      Example: 3840x2160 -> 1280x720.
    - Small videos are optionally upscaled to at least target_h x target_w.
    - The final size is rounded to a multiple of size_multiple.
    """
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid original size: {orig_w}x{orig_h}")

    need_downscale = orig_h > target_h or orig_w > target_w
    need_upscale = upscale_small_videos and (orig_h < target_h or orig_w < target_w)

    if need_downscale:
        scale = min(target_h / float(orig_h), target_w / float(orig_w))
        scaled_h = orig_h * scale
        scaled_w = orig_w * scale
        max_h = int(math.floor(target_h / size_multiple) * size_multiple)
        max_w = int(math.floor(target_w / size_multiple) * size_multiple)
        out_h = clamp_round_to_multiple(scaled_h, size_multiple, max_x=max_h)
        out_w = clamp_round_to_multiple(scaled_w, size_multiple, max_x=max_w)
    elif need_upscale:
        scale = max(target_h / float(orig_h), target_w / float(orig_w))
        scaled_h = orig_h * scale
        scaled_w = orig_w * scale
        out_h = ceil_to_multiple(int(round(scaled_h)), size_multiple)
        out_w = ceil_to_multiple(int(round(scaled_w)), size_multiple)
    else:
        out_h = clamp_round_to_multiple(orig_h, size_multiple)
        out_w = clamp_round_to_multiple(orig_w, size_multiple)

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


def rgb_array_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame)


def has_mask(mask_frame: np.ndarray) -> bool:
    if mask_frame.ndim == 3:
        return bool(mask_frame.max() > 0)
    return bool(np.max(mask_frame) > 0)


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


def get_video_meta(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, height, width


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap


def read_next_chunk(
    video_cap: cv2.VideoCapture,
    mask_cap: cv2.VideoCapture,
    chunk_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    video_chunk: List[np.ndarray] = []
    mask_chunk: List[np.ndarray] = []

    for _ in range(chunk_size):
        v_ok, v_frame = video_cap.read()
        m_ok, m_frame = mask_cap.read()

        if v_ok != m_ok:
            raise RuntimeError("Video and mask ended at different times.")
        if not v_ok:
            break

        video_chunk.append(cv2.cvtColor(v_frame, cv2.COLOR_BGR2RGB))
        mask_chunk.append(cv2.cvtColor(m_frame, cv2.COLOR_BGR2RGB))

    return video_chunk, mask_chunk


def find_first_valid_ref_crop(video_path: str, mask_path: str) -> torch.Tensor:
    video_cap = open_video_capture(video_path)
    mask_cap = open_video_capture(mask_path)
    try:
        while True:
            v_ok, v_frame = video_cap.read()
            m_ok, m_frame = mask_cap.read()
            if v_ok != m_ok:
                raise RuntimeError("Video and mask ended at different times while searching reference crop.")
            if not v_ok:
                break

            video_rgb = cv2.cvtColor(v_frame, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(m_frame, cv2.COLOR_BGR2RGB)
            if has_mask(mask_rgb):
                return crop_square_from_pil(
                    rgb_array_to_pil(mask_rgb),
                    rgb_array_to_pil(video_rgb),
                    target_size=224,
                    video_mask_path=mask_path,
                )
    finally:
        video_cap.release()
        mask_cap.release()

    raise ValueError(f"No valid mask region found in entire video: {mask_path}")


def open_video_writer(output_path: Path, fps: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    writer = imageio.get_writer(
        tmp_path.as_posix(),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        quality=8,
    )
    return writer, tmp_path


def detect_frame_length_arg(pipe) -> Optional[str]:
    try:
        sig = inspect.signature(pipe.__call__)
    except Exception:
        return None

    candidate_names = ["num_frames", "video_length", "frames", "num_video_frames"]
    for name in candidate_names:
        if name in sig.parameters:
            return name
    return None


def build_pipeline(args, device_str: str):
    model_manager = ModelManager(device=device_str)
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
        device=device_str,
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)
    frame_length_arg = detect_frame_length_arg(pipe)
    return pipe, frame_length_arg


def run_single_sample(
    pipe: WanRemovePipeline,
    frame_length_arg: Optional[str],
    args,
    sample_dir: Path,
    device: torch.device,
    log_prefix: str,
) -> None:
    video_path = str(sample_dir / "video.mp4")
    mask_path = str(sample_dir / "mask.mp4")
    output_path = build_output_path(sample_dir, args.input_root, args.output_root, args.output_name)

    if output_path.exists() and not args.overwrite:
        print(f"{log_prefix} [SKIP] Output exists: {output_path}")
        return

    if output_path.exists() and args.overwrite:
        output_path.unlink()

    print(f"\n{log_prefix} [INFO] Processing sample: {sample_dir}")
    start_t = time.time()

    video_frame_count, fps, orig_h, orig_w = get_video_meta(video_path)
    mask_frame_count, mask_fps, _, _ = get_video_meta(mask_path)
    if video_frame_count != mask_frame_count:
        raise ValueError(
            f"Frame count mismatch between video and mask: {video_path} ({video_frame_count}) vs {mask_path} ({mask_frame_count})"
        )

    target_h, target_w = get_infer_size(
        orig_h=orig_h,
        orig_w=orig_w,
        target_h=args.target_height,
        target_w=args.target_width,
        size_multiple=args.size_multiple,
        upscale_small_videos=args.upscale_small_videos,
    )

    print(
        f"{log_prefix} [INFO] video frames={video_frame_count}, fps={fps:.4f}, "
        f"original={orig_w}x{orig_h}, target={target_w}x{target_h}, chunk_size={args.chunk_size}"
    )
    if abs(fps - mask_fps) > 1e-3:
        print(
            f"{log_prefix} [WARN] video fps ({fps:.4f}) != mask fps ({mask_fps:.4f}); output will follow source video fps."
        )

    global_ref_crop = find_first_valid_ref_crop(video_path, mask_path)

    video_cap = open_video_capture(video_path)
    mask_cap = open_video_capture(mask_path)
    writer, tmp_output_path = open_video_writer(output_path, fps)

    total_written = 0
    try:
        chunk_index = 0
        while True:
            video_chunk, mask_chunk = read_next_chunk(video_cap, mask_cap, args.chunk_size)
            if len(video_chunk) == 0:
                break

            valid_len = len(video_chunk)
            start_frame = chunk_index * args.chunk_size
            end_frame = start_frame + valid_len

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
                chunk_size=args.chunk_size,
                is_mask=False,
            ).to(device)
            mask_tensor = prepare_chunk_tensor(
                frames=mask_chunk,
                target_h=target_h,
                target_w=target_w,
                chunk_size=args.chunk_size,
                is_mask=True,
            ).to(device)
            ref_crop = ref_crop.to(device)

            pipe_kwargs = dict(
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
            if frame_length_arg is not None:
                pipe_kwargs[frame_length_arg] = args.chunk_size

            print(
                f"{log_prefix} [INFO] chunk {start_frame:05d}:{end_frame:05d} "
                f"(valid={valid_len}, padded_to={args.chunk_size})"
            )
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                remove_video, _ = pipe(**pipe_kwargs)

            remove_video = np.asarray(remove_video)
            if remove_video.ndim != 4:
                raise ValueError(f"Unexpected model output shape: {remove_video.shape}")
            if len(remove_video) < valid_len:
                raise ValueError(
                    f"Model output shorter than valid chunk length: got {len(remove_video)}, need {valid_len}"
                )

            for frame in remove_video[:valid_len]:
                writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
                total_written += 1

            del video_tensor, mask_tensor, ref_crop, remove_video
            torch.cuda.empty_cache()
            chunk_index += 1

        if total_written != video_frame_count:
            raise RuntimeError(
                f"Output frame count mismatch for {sample_dir}: output={total_written} vs input={video_frame_count}"
            )
    except Exception:
        writer.close()
        video_cap.release()
        mask_cap.release()
        if tmp_output_path.exists():
            try:
                tmp_output_path.unlink()
            except Exception:
                pass
        raise
    else:
        writer.close()
        video_cap.release()
        mask_cap.release()
        os.replace(tmp_output_path, output_path)

    print(
        f"{log_prefix} [INFO] Done sample: {sample_dir.name} | frames={video_frame_count} | cost={time.time() - start_t:.2f}s"
    )


def worker_main(worker_rank: int, gpu_id: int, sample_dirs: List[str], args, result_queue=None):
    log_prefix = f"[GPU {gpu_id} | worker {worker_rank}]"
    if not sample_dirs:
        print(f"{log_prefix} [INFO] No samples assigned.")
        if result_queue is not None:
            result_queue.put({"gpu_id": gpu_id, "ok": 0, "fail": 0, "failed_samples": []})
        return

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    device_str = f"cuda:{gpu_id}"

    print(f"{log_prefix} [INFO] Assigned {len(sample_dirs)} samples.")
    print(f"{log_prefix} [INFO] Building model on {device_str}...")
    pipe, frame_length_arg = build_pipeline(args, device_str=device_str)
    if frame_length_arg is not None:
        print(f"{log_prefix} [INFO] Detected pipeline frame-length argument: {frame_length_arg}")

    num_ok = 0
    num_fail = 0
    failed_samples = []

    for sample_dir_str in sample_dirs:
        sample_dir = Path(sample_dir_str)
        try:
            run_single_sample(
                pipe=pipe,
                frame_length_arg=frame_length_arg,
                args=args,
                sample_dir=sample_dir,
                device=device,
                log_prefix=log_prefix,
            )
            num_ok += 1
        except Exception as e:
            num_fail += 1
            failed_samples.append((str(sample_dir), str(e)))
            print(f"{log_prefix} [ERROR] Failed sample: {sample_dir}")
            print(f"{log_prefix} [ERROR] {e}")
            torch.cuda.empty_cache()

    print(f"{log_prefix} [INFO] Finished. success={num_ok}, fail={num_fail}")
    if result_queue is not None:
        result_queue.put(
            {
                "gpu_id": gpu_id,
                "ok": num_ok,
                "fail": num_fail,
                "failed_samples": failed_samples,
            }
        )


def main():
    args = build_parser().parse_args()

    sample_dirs = list_sample_dirs(args.input_root, args.recursive)
    if not sample_dirs:
        raise RuntimeError(
            f"No sample folders found under {args.input_root}. Each sample must contain video.mp4 and mask.mp4."
        )

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpu_ids is empty.")
    if max(gpu_ids) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested gpu_ids={gpu_ids}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )

    print(f"[INFO] Found {len(sample_dirs)} sample folders.")
    print(f"[INFO] Using GPUs: {gpu_ids}")
    print(f"[INFO] 4K-like large videos will be scaled down to fit within {args.target_width}x{args.target_height} for inference.")

    sample_dirs_str = [str(p) for p in sample_dirs]
    world_size = len(gpu_ids)
    shards = [sample_dirs_str[i::world_size] for i in range(world_size)]

    if world_size == 1:
        worker_main(worker_rank=0, gpu_id=gpu_ids[0], sample_dirs=shards[0], args=args, result_queue=None)
        return

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(target=worker_main, args=(rank, gpu_id, shards[rank], args, result_queue))
        p.start()
        procs.append(p)

    summaries = []
    for _ in procs:
        summaries.append(result_queue.get())

    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"[WARN] A worker exited with code {p.exitcode}.")

    total_ok = sum(x["ok"] for x in summaries)
    total_fail = sum(x["fail"] for x in summaries)

    print("\n========== SUMMARY ==========")
    print(f"[INFO] Success: {total_ok}")
    print(f"[INFO] Failed : {total_fail}")
    failed_samples = []
    for summary in summaries:
        failed_samples.extend(summary["failed_samples"])
    if failed_samples:
        print("[INFO] Failed samples:")
        for path, err in failed_samples:
            print(f"  - {path}: {err}")


if __name__ == "__main__":
    main()
