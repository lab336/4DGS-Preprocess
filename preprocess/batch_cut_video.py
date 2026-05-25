"""
批量提取视频帧脚本（多进程加速版）

输出目录结构：
  output/
  ├── by_video/          # 同一视频的帧放在同一子文件夹（按帧序号 1,2,3... 命名）
  │   ├── video_name/
  │   │   ├── 1.jpg
  │   │   └── ...
  │   └── ...
  └── by_frame/          # 同一帧序号的图片放在同一子文件夹（按视频序号 1,2,3... 命名）
      ├── 1/
      │   ├── 1.jpg
      │   └── ...
      └── ...

超参数：
  --data_dir    : 视频所在目录，默认 video
  --output_dir  : 输出目录，默认 output
  --num_frames  : 每个视频提取多少帧，默认 0（0 表示全部）
  --target_fps  : 目标帧率，如视频 120fps 设为 30 则每秒取 30 帧（默认 0 = 不限）
  --sample_mode : 采样方式：uniform / head（默认 uniform）
  --ext         : 视频文件扩展名过滤，默认 .mp4
  --output_mode : 输出组织方式：by_video / by_frame / both（默认 by_frame）
  --workers     : 并行进程数，默认 CPU 核数 / 2
  --use_seek    : 启用 seek 快速跳帧（稀疏采样时极大提速，默认关闭）
  --start_idx   : 视频/摄像头编号的起始值（默认 1，用于多批次续接编号）

优先级：先按 target_fps 降采样，再按 num_frames 限制总帧数。
"""

import os
import argparse
import cv2
import numpy as np
import multiprocessing

# 抑制 GStreamer / OpenCV 无关警告
os.environ["GST_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ─────────────────────────── 工具函数 ───────────────────────────

def get_sample_indices(total_frames: int, num_frames: int, sample_mode: str,
                       original_fps: float = 0, target_fps: float = 0):
    """
    计算要提取的帧索引（0-based 有序列表），或 None 表示全部帧。
    处理顺序：先按 target_fps 做时间降采样，再按 num_frames 限制总数。
    """
    # ── Step 1：按 target_fps 降采样 ──────────────────────────────────
    if target_fps > 0 and original_fps > 0 and target_fps < original_fps:
        step = original_fps / target_fps
        candidates = []
        pos = 0.0
        while True:
            idx = int(round(pos))
            if idx >= total_frames:
                break
            candidates.append(idx)
            pos += step
        # 去重（浮点舍入可能产生相邻重复）
        indices = sorted(set(candidates))
    else:
        indices = None  # None = 全部帧

    # ── Step 2：按 num_frames 截取 ────────────────────────────────────
    if num_frames > 0:
        pool = indices if indices is not None else list(range(total_frames))
        if num_frames >= len(pool):
            return indices  # 已经足够少
        if sample_mode == "head":
            return pool[:num_frames]
        # uniform
        sel = np.linspace(0, len(pool) - 1, num_frames, dtype=int)
        return [pool[i] for i in sel]

    return indices


def open_video(video_path: str):
    """尝试用 FFMPEG 后端打开，失败则用默认后端。"""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
        return cap
    cap.release()
    return cv2.VideoCapture(video_path)


def _probe_video(video_path: str):
    """快速探测视频的帧数和帧率（不解码）。"""
    cap = open_video(video_path)
    if not cap.isOpened():
        return 0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


def _write_safe(path: str, frame) -> None:
    """写盘辅助：异常时打印警告而非崩溃整个进程。"""
    try:
        cv2.imwrite(path, frame)
    except Exception as exc:
        print(f"\n  [警告] 写盘失败 {path}: {exc}", flush=True)


# ─────────────────────────── 单视频处理（多进程 worker） ───────────────────────────

def process_video(task: dict) -> int:
    """
    处理一个视频：解码 + 写盘。
    速度优化：
      - 不需要的帧只调 grab()（仅 demux，跳过解码），需要的帧调 grab()+retrieve()。
      - seek 模式直接定位目标帧，跳过大段无用数据。
    """
    video_path   = task["video_path"]
    video_idx    = task["video_idx"]
    bv_subdir    = task["bv_subdir"]
    by_frame_dir = task["by_frame_dir"]
    num_frames   = task["num_frames"]
    target_fps   = task["target_fps"]
    sample_mode  = task["sample_mode"]
    total_videos = task["total_videos"]
    output_mode  = task["output_mode"]
    use_seek     = task["use_seek"]
    start_idx    = task.get("start_idx", 1)

    try:
        cap = open_video(video_path)
        if not cap.isOpened():
            print(f"  [跳过] 无法打开：{os.path.basename(video_path)}", flush=True)
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        indices = get_sample_indices(total, num_frames, sample_mode, fps, target_fps)
        actual  = total if indices is None else len(indices)

        extra = ""
        if target_fps > 0 and fps > 0:
            extra = f"  原始fps={fps:.1f}  目标fps={target_fps}"
        print(f"[{video_idx:03d}/{total_videos}] {os.path.basename(video_path)}"
              f"  总帧数={total}  提取={actual}{extra}", flush=True)

        need_bv      = output_mode in ("by_video", "both")
        need_bf      = output_mode in ("by_frame", "both")
        report_every = max(1, actual // 20)
        frame_count  = 0

        # ── Seek 模式：直接跳到目标帧 ────────────────────────────────────
        if use_seek and indices is not None:
            for frame_seq, raw_idx in enumerate(indices, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(raw_idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                if need_bv:
                    _write_safe(os.path.join(bv_subdir, f"{frame_seq + start_idx - 1}.jpg"), frame)
                if need_bf:
                    _write_safe(os.path.join(by_frame_dir, str(frame_seq),
                                             f"{video_idx}.jpg"), frame)
                frame_count = frame_seq
                if frame_seq % report_every == 0 or frame_seq == actual:
                    print(f"  [{video_idx:03d}] 进度：{frame_seq}/{actual} 帧",
                          end="\r", flush=True)

        # ── 顺序模式：grab() 跳帧 + retrieve() 解码目标帧 ───────────────
        else:
            sample_set = set(indices) if indices is not None else None
            raw_idx    = 0
            # head 模式的最大原始索引，用于提前退出
            max_raw = max(indices) if indices is not None else total - 1
            while True:
                if sample_set is not None and frame_count >= len(sample_set):
                    break

                need_this_frame = (sample_set is None or raw_idx in sample_set)

                if need_this_frame:
                    # grab + retrieve：完整解码
                    if not cap.grab():
                        break
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    frame_count += 1
                    if need_bv:
                        _write_safe(os.path.join(bv_subdir, f"{frame_count + start_idx - 1}.jpg"), frame)
                    if need_bf:
                        _write_safe(os.path.join(by_frame_dir, str(frame_count),
                                                 f"{video_idx}.jpg"), frame)
                    if frame_count % report_every == 0 or frame_count == actual:
                        print(f"  [{video_idx:03d}] 进度：{frame_count}/{actual} 帧",
                              end="\r", flush=True)
                else:
                    # grab() 仅 demux，不做完整解码，比 read() 快很多
                    if not cap.grab():
                        break

                # 已经超过最后需要的帧，提前退出
                if raw_idx >= max_raw:
                    break
                raw_idx += 1

        cap.release()
        print(f"  [{video_idx:03d}] 完成：共写出 {frame_count} 帧{' ' * 20}", flush=True)
        return frame_count

    except Exception as exc:
        print(f"\n  [{video_idx:03d}] 错误：{exc}", flush=True)
        import traceback
        traceback.print_exc()
        return 0


# ─────────────────────────── 主入口 ───────────────────────────

def main():
    cpu_count       = multiprocessing.cpu_count()
    default_workers = max(1, cpu_count // 2)

    parser = argparse.ArgumentParser(description="批量提取视频帧（多进程加速版）")
    parser.add_argument("--data_dir",    type=str, default="DJI/video/3.mp4",
                        help="视频所在目录")
    parser.add_argument("--start_idx",   type=int, default=1,
                        help="视频/摄像头编号的起始值（默认 1，可设为其他值以续接之前的批次）")
    parser.add_argument("--output_dir",  type=str, default="DJI/images",
                        help="输出根目录")
    parser.add_argument("--num_frames",  type=int, default=0,
                        help="每视频提取帧数（0=全部）")
    parser.add_argument("--target_fps",  type=float, default=3,
                        help="目标帧率（如原始120fps设30则每秒取30帧，0=不限）")
    parser.add_argument("--sample_mode", type=str, default="uniform",
                        choices=["uniform", "head"],
                        help="采样方式：uniform=均匀采样，head=严格前N帧")
    parser.add_argument("--ext",         type=str, default=".mp4",
                        help="视频文件扩展名")
    parser.add_argument("--output_mode", type=str, default="by_video",
                        choices=["by_video", "by_frame", "both"],
                        help="输出组织方式：by_video / by_frame / both（默认 by_video）")
    parser.add_argument("--workers",     type=int, default=default_workers,
                        help=f"并行处理视频的进程数（默认 {default_workers}）")
    parser.add_argument("--use_seek",    action="store_true",
                        help="启用 seek 快速跳帧（稀疏采样时显著提速，默认关闭）")
    args = parser.parse_args()

    data_dir   = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    # 支持单个视频文件作为输入
    if os.path.isfile(data_dir):
        video_path = data_dir
        data_dir   = os.path.dirname(video_path)
        video_files = [os.path.basename(video_path)]
    else:
        video_files = sorted(
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() == args.ext.lower()
        )
    if not video_files:
        print(f"在 {data_dir} 中未找到 {args.ext} 视频文件")
        return

    by_video_dir = os.path.join(output_dir, "by_video")
    by_frame_dir = os.path.join(output_dir, "by_frame")
    if args.output_mode in ("by_video", "both"):
        os.makedirs(by_video_dir, exist_ok=True)
    if args.output_mode in ("by_frame", "both"):
        os.makedirs(by_frame_dir, exist_ok=True)

    # ── 预建所有子目录（主进程统一创建，worker 内不调用 makedirs）──────────
    if args.output_mode in ("by_video", "both"):
        for filename in video_files:
            os.makedirs(os.path.join(by_video_dir, os.path.splitext(filename)[0]),
                        exist_ok=True)

    if args.output_mode in ("by_frame", "both"):
        print("预扫描视频帧数以提前建立目录...")
        max_frames = 0
        for filename in video_files:
            total, fps = _probe_video(os.path.join(data_dir, filename))
            if total > 0:
                indices = get_sample_indices(total, args.num_frames, args.sample_mode,
                                             fps, args.target_fps)
                needed = total if indices is None else len(indices)
                max_frames = max(max_frames, needed)
        print(f"预建 by_frame 子目录（共 {max_frames} 个）...")
        for i in range(1, max_frames + 1):
            os.makedirs(os.path.join(by_frame_dir, str(i)), exist_ok=True)

    # ── 构造任务列表 ───────────────────────────────────────────────────────────
    tasks = [
        {
            "video_path":   os.path.join(data_dir, filename),
            "video_idx":    idx,
            "video_name":   os.path.splitext(filename)[0],
            "bv_subdir":    os.path.join(by_video_dir, os.path.splitext(filename)[0]),
            "by_frame_dir": by_frame_dir,
            "num_frames":   args.num_frames,
            "target_fps":   args.target_fps,
            "sample_mode":  args.sample_mode,
            "total_videos": len(video_files),
            "output_mode":  args.output_mode,
            "use_seek":     args.use_seek,
            "start_idx":    args.start_idx,
        }
        for idx, filename in enumerate(video_files, start=1)
    ]

    workers = min(args.workers, len(video_files))
    print(f"共找到 {len(video_files)} 个视频，启用 {workers} 进程并行处理...\n")

    if workers <= 1:
        for task in tasks:
            process_video(task)
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            pool.map(process_video, tasks)

    print("\n全部完成！")
    if args.output_mode in ("by_video", "both"):
        print(f"  按视频组织：{by_video_dir}")
    if args.output_mode in ("by_frame", "both"):
        print(f"  按帧序号组织：{by_frame_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 打包为 exe 时需要
    main()

