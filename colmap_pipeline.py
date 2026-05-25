"""
固定多相机阵列 COLMAP 动态多视图重建预处理流程

数据集结构:
  dataset/
    frame_000/
      0.png, 1.png, ..., n.png
    frame_001/
      ...
    frame_m/

输出结构:
  output/
    sparse/0/cameras.txt, images.txt, points3D.txt
    undistorter_images/1/, 2/, ..., m/  (每个文件夹下 1.png, 2.png, ..., n.png)
    points_cloud/1.ply, 2.ply, ..., m.ply

# 完整流程（标定 + 所有帧 MVS）
python colmap_pipeline.py --dataset ./dataset --output ./output --calib_frame 0

# 仅标定（不跑 MVS）
python colmap_pipeline.py --dataset ./dataset --output ./output --calib_frame 0 --skip_mvs

# 跳过标定，只跑 MVS（已有 sparse/0）
python colmap_pipeline.py --dataset ./dataset --output ./output --skip_calibration

# 处理指定帧范围（0-based 索引）
python colmap_pipeline.py --dataset ./dataset --output ./output --skip_calibration --frame_range 0:9

# 禁用 GPU
python colmap_pipeline.py --dataset ./dataset --output ./output --calib_frame 0 --no_gpu

# 调试模式（保留临时文件 + 详细日志）
python colmap_pipeline.py --dataset ./dataset --output ./output --calib_frame 0 --keep_workspace --verbose
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("请先安装 tqdm: pip install tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def find_colmap() -> str:
    """检测 COLMAP 是否可用，返回可执行文件路径。"""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        logger.error("未找到 COLMAP，请确保 colmap 已安装并加入 PATH。")
        sys.exit(1)
    logger.info(f"找到 COLMAP: {colmap_path}")
    return colmap_path


def run_colmap(colmap: str, args: list, description: str = "") -> bool:
    """调用 COLMAP 子命令，返回是否成功。"""
    cmd = [colmap] + args
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"执行: {cmd_str}")
    if description:
        logger.info(f"  -> {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            logger.debug(f"  [stdout] {line}")
    if result.returncode != 0:
        logger.error(f"命令失败 (exit={result.returncode}): {cmd_str}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-30:]:
                logger.error(f"  [stderr] {line}")
        return False
    logger.info(f"  完成 ✓")
    return True


def discover_frames(dataset_dir: Path) -> list:
    """发现所有帧文件夹，按数字顺序排序返回（文件夹名为纯数字时按数值排序）。"""

    def sort_key(p: Path):
        return (int(p.name), p.name) if p.name.isdigit() else (float("inf"), p.name)

    frames = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir()],
        key=sort_key,
    )
    if not frames:
        logger.error(f"在 {dataset_dir} 中未找到任何帧文件夹。")
        sys.exit(1)
    logger.info(f"发现 {len(frames)} 帧: {frames[0].name} ... {frames[-1].name}")
    return frames


def phase1_calibration(
    colmap: str,
    calib_frame_dir: Path,
    output_dir: Path,
    local_workspace_dir: Path,
    max_image_size: int,
    max_num_features: int,
    use_gpu: bool,
) -> bool:
    """
    第一阶段：在指定帧上运行完整 SfM 流程，恢复相机参数。
    COLMAP 工作文件写入 local_workspace_dir（本地磁盘），最终 TXT 模型复制到 output_dir。
    """
    logger.info("=" * 60)
    logger.info(f"第一阶段：自动标定 (使用 {calib_frame_dir.name})")
    logger.info("=" * 60)

    work_dir = local_workspace_dir / "_calibration_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    # 1. database_creator
    if db_path.exists():
        db_path.unlink()
    if not run_colmap(
        colmap,
        ["database_creator", "--database_path", str(db_path)],
        "创建数据库",
    ):
        return False

    # 2. feature_extractor
    if not run_colmap(
        colmap,
        [
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(calib_frame_dir),
            "--ImageReader.single_camera", "0",
            "--FeatureExtraction.use_gpu", gpu_flag,
            "--FeatureExtraction.max_image_size", str(max_image_size),
            "--SiftExtraction.max_num_features", str(max_num_features),
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ],
        "特征提取",
    ):
        return False

    # 3. exhaustive_matcher
    if not run_colmap(
        colmap,
        [
            "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
        ],
        "穷举匹配",
    ):
        return False

    # 4. mapper
    if not run_colmap(
        colmap,
        [
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(calib_frame_dir),
            "--output_path", str(sparse_dir),
        ],
        "稀疏重建 (mapper)",
    ):
        return False

    # 检查 sparse/0 是否存在
    sparse_model = sparse_dir / "0"
    if not sparse_model.exists():
        logger.error(f"mapper 输出目录 {sparse_model} 不存在，标定失败。")
        return False

    # 5. model_converter -> TXT
    final_sparse = output_dir / "sparse" / "0"
    final_sparse.mkdir(parents=True, exist_ok=True)

    if not run_colmap(
        colmap,
        [
            "model_converter",
            "--input_path", str(sparse_model),
            "--output_path", str(final_sparse),
            "--output_type", "TXT",
        ],
        "导出 TXT 格式模型",
    ):
        return False

    # 验证输出
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        if not (final_sparse / fname).exists():
            logger.error(f"缺少 {fname}，标定失败。")
            return False

    logger.info(f"标定完成，模型已保存到 {final_sparse}")
    return True


def build_images_txt_for_frame(
    ref_images_txt: Path, frame_image_dir: Path, out_images_txt: Path
):
    """
    根据标定帧的 images.txt，生成当前帧的 images.txt。
    保持相机位姿不变，只需确保图像文件名与当前帧图像对应。
    由于每帧图像文件名相同（0.png, 1.png, ...），直接复制即可。
    """
    shutil.copy2(ref_images_txt, out_images_txt)


def phase2_mvs_single_frame(
    colmap: str,
    frame_dir: Path,
    frame_idx: int,
    ref_sparse_dir: Path,
    output_dir: Path,
    local_workspace_dir: Path,
    use_gpu: bool,
    geom_consistency: bool,
    fusion_min_num_pixels: int,
    fusion_check_num_images: int,
) -> bool:
    """
    第二阶段：对单帧执行 MVS（undistort + patch_match + fusion）。
    COLMAP 工作文件写入 local_workspace_dir（本地磁盘），最终结果复制到 output_dir。
    """
    gpu_flag = "1" if use_gpu else "0"

    # 工作目录：临时 dense workspace（必须在本地磁盘，COLMAP 不支持 UNC 网络路径）
    work_dir = local_workspace_dir / "_mvs_workspace" / str(frame_idx)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 准备 sparse 模型（复制标定结果）
    frame_sparse = work_dir / "sparse_input" / "0"
    frame_sparse.mkdir(parents=True, exist_ok=True)
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        src = ref_sparse_dir / fname
        dst = frame_sparse / fname
        shutil.copy2(src, dst)

    # 先将 TXT 转为 BIN（image_undistorter 需要 BIN 格式）
    frame_sparse_bin = work_dir / "sparse_bin" / "0"
    frame_sparse_bin.mkdir(parents=True, exist_ok=True)
    if not run_colmap(
        colmap,
        [
            "model_converter",
            "--input_path", str(frame_sparse),
            "--output_path", str(frame_sparse_bin),
            "--output_type", "BIN",
        ],
        f"帧 {frame_idx}: TXT -> BIN 转换",
    ):
        return False

    # 1. image_undistorter
    dense_dir = work_dir / "dense"
    if not run_colmap(
        colmap,
        [
            "image_undistorter",
            "--image_path", str(frame_dir),
            "--input_path", str(frame_sparse_bin),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ],
        f"帧 {frame_idx}: 图像去畸变",
    ):
        return False

    # 复制去畸变后的图像到 undistorter_images/<frame_idx>/
    undist_src = dense_dir / "images"
    undist_dst = output_dir / "undistorter_images" / str(frame_idx)
    undist_dst.mkdir(parents=True, exist_ok=True)

    if undist_src.exists():
        # 按文件名排序，重命名为 1.png, 2.png, ...
        src_images = sorted(undist_src.iterdir(), key=lambda p: p.stem)
        for i, img in enumerate(src_images, start=1):
            shutil.copy2(img, undist_dst / f"{i}{img.suffix}")
        logger.info(
            f"帧 {frame_idx}: 已复制 {len(src_images)} 张去畸变图像到 {undist_dst}"
        )
    else:
        logger.warning(f"帧 {frame_idx}: 去畸变图像目录 {undist_src} 不存在")

    # 2. patch_match_stereo
    if not run_colmap(
        colmap,
        [
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true" if geom_consistency else "false",
            "--PatchMatchStereo.gpu_index", "0" if use_gpu else "-1",
        ],
        f"帧 {frame_idx}: PatchMatch 立体匹配",
    ):
        return False

    # 3. stereo_fusion（先写到本地，再复制到网络输出目录）
    local_ply_path = work_dir / f"{frame_idx}.ply"

    if not run_colmap(
        colmap,
        [
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric" if geom_consistency else "photometric",
            "--output_path", str(local_ply_path),
            "--StereoFusion.check_num_images", str(fusion_check_num_images),
            "--StereoFusion.min_num_pixels", str(fusion_min_num_pixels),
        ],
        f"帧 {frame_idx}: 点云融合",
    ):
        return False

    # 复制 PLY 到最终输出目录
    ply_dst_dir = output_dir / "points_cloud"
    ply_dst_dir.mkdir(parents=True, exist_ok=True)
    ply_dst = ply_dst_dir / f"{frame_idx}.ply"
    shutil.copy2(local_ply_path, ply_dst)
    logger.info(f"帧 {frame_idx}: 点云已保存到 {ply_dst}")
    return True


def cleanup_mvs_workspace(local_workspace_dir: Path, keep: bool = False):
    """清理本地临时工作目录。"""
    for sub in ["_calibration_workspace", "_mvs_workspace"]:
        ws = local_workspace_dir / sub
        if ws.exists() and not keep:
            logger.info(f"清理临时目录: {ws}")
            shutil.rmtree(ws, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="固定多相机阵列 COLMAP 动态多视图重建预处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        help="数据集根目录，包含 frame_000, frame_001, ... 等子文件夹",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出根目录",
    )
    parser.add_argument(
        "--calib_frame", type=str, default=None,
        help=(
            "用于标定的帧文件夹名称（如 '120' 表示使用名为 '120' 的文件夹）。"
            "也可以传入 0-based 整数索引（如 '0' 表示第一帧）。"
            "默认使用第一帧。"
        ),
    )
    parser.add_argument(
        "--skip_calibration", action="store_true",
        help="跳过标定阶段（假设 output/sparse/0/ 已存在）",
    )
    parser.add_argument(
        "--skip_mvs", action="store_true",
        help="只运行标定，不执行 MVS",
    )
    parser.add_argument(
        "--frame_range", type=str, default=None,
        help="指定要处理的帧范围，格式: start:end (含两端, 0-based)，默认处理所有帧",
    )
    parser.add_argument(
        "--max_image_size", type=int, default=2400,
        help="特征提取最大图像尺寸 (默认 2400)",
    )
    parser.add_argument(
        "--max_num_features", type=int, default=16384,
        help="SIFT 最大特征数 (默认 16384)",
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="禁用 GPU",
    )
    parser.add_argument(
        "--geom_consistency", action="store_true", default=True,
        help="PatchMatch 使用几何一致性 (默认开启)",
    )
    parser.add_argument(
        "--no_geom_consistency", action="store_true",
        help="PatchMatch 不使用几何一致性",
    )
    parser.add_argument(
        "--fusion_min_num_pixels", type=int, default=3,
        help="融合最小像素数 (默认 3)",
    )
    parser.add_argument(
        "--fusion_check_num_images", type=int, default=2,
        help="融合检查图像数 (默认 2)",
    )
    parser.add_argument(
        "--local_workspace", type=str, default=None,
        help=(
            "COLMAP 临时工作目录（必须在本地磁盘，不能是 UNC 网络路径）。"
            "默认使用系统临时目录下的 colmap_pipeline_work 子目录。"
        ),
    )
    parser.add_argument(
        "--keep_workspace", action="store_true",
        help="保留临时工作目录（用于调试）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="显示详细日志",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.no_geom_consistency:
        args.geom_consistency = False

    use_gpu = not args.no_gpu
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()

    if not dataset_dir.exists():
        logger.error(f"数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 本地工作目录（COLMAP 不支持 UNC 网络路径）
    if args.local_workspace:
        local_workspace_dir = Path(args.local_workspace).resolve()
    else:
        local_workspace_dir = Path(tempfile.gettempdir()) / "colmap_pipeline_work"
    local_workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"COLMAP 本地工作目录: {local_workspace_dir}")

    # 检测 COLMAP
    colmap = find_colmap()

    # 发现所有帧
    frames = discover_frames(dataset_dir)
    n_frames = len(frames)

    # ====================== 第一阶段：标定 ======================
    ref_sparse_dir = output_dir / "sparse" / "0"

    if not args.skip_calibration:
        # 解析标定帧：优先按文件夹名查找，找不到则按 0-based 索引
        calib_frame_dir = None
        if args.calib_frame is not None:
            # 先按文件夹名精确匹配
            for f in frames:
                if f.name == args.calib_frame:
                    calib_frame_dir = f
                    break
            # 再尝试作为 0-based 整数索引
            if calib_frame_dir is None and args.calib_frame.isdigit():
                idx = int(args.calib_frame)
                # 若该数字恰好是某个文件夹名（已在上面查过），这里是索引语义
                if 0 <= idx < n_frames:
                    calib_frame_dir = frames[idx]
                    logger.info(
                        f"未找到名为 '{args.calib_frame}' 的文件夹，"
                        f"按 0-based 索引使用帧: {calib_frame_dir.name}"
                    )
            if calib_frame_dir is None:
                logger.error(
                    f"无法找到标定帧 '{args.calib_frame}'，"
                    f"请检查文件夹名称或索引范围 [0, {n_frames - 1}]。"
                )
                sys.exit(1)
        else:
            calib_frame_dir = frames[0]
        logger.info(f"标定帧: {calib_frame_dir}")

        success = phase1_calibration(
            colmap=colmap,
            calib_frame_dir=calib_frame_dir,
            output_dir=output_dir,
            local_workspace_dir=local_workspace_dir,
            max_image_size=args.max_image_size,
            max_num_features=args.max_num_features,
            use_gpu=use_gpu,
        )
        if not success:
            logger.error("第一阶段（标定）失败，退出。")
            sys.exit(1)
    else:
        if not ref_sparse_dir.exists():
            logger.error(
                f"跳过标定但 {ref_sparse_dir} 不存在，请先运行标定。"
            )
            sys.exit(1)
        logger.info(f"跳过标定，使用已有模型: {ref_sparse_dir}")

    # 验证标定输出
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        if not (ref_sparse_dir / fname).exists():
            logger.error(f"缺少标定文件 {ref_sparse_dir / fname}")
            sys.exit(1)

    if args.skip_mvs:
        logger.info("仅标定模式，跳过 MVS 阶段。")
        return

    # ====================== 第二阶段：MVS ======================
    logger.info("=" * 60)
    logger.info("第二阶段：固定相机参数处理所有帧 (MVS)")
    logger.info("=" * 60)

    # 解析帧范围
    if args.frame_range:
        parts = args.frame_range.split(":")
        start_idx = int(parts[0])
        end_idx = int(parts[1])
        frame_indices = list(range(start_idx, end_idx + 1))
    else:
        frame_indices = list(range(n_frames))

    success_count = 0
    fail_count = 0
    failed_frames = []

    for idx in tqdm(frame_indices, desc="MVS 处理帧", unit="帧"):
        if idx < 0 or idx >= n_frames:
            logger.warning(f"帧索引 {idx} 超出范围，跳过。")
            continue

        frame_dir = frames[idx]
        frame_number = idx + 1  # 1-based 用于输出命名
        logger.info(f"\n{'─' * 40}")
        logger.info(f"处理帧 {frame_number}/{n_frames}: {frame_dir.name}")
        logger.info(f"{'─' * 40}")

        try:
            ok = phase2_mvs_single_frame(
                colmap=colmap,
                frame_dir=frame_dir,
                frame_idx=frame_number,
                ref_sparse_dir=ref_sparse_dir,
                output_dir=output_dir,
                local_workspace_dir=local_workspace_dir,
                use_gpu=use_gpu,
                geom_consistency=args.geom_consistency,
                fusion_min_num_pixels=args.fusion_min_num_pixels,
                fusion_check_num_images=args.fusion_check_num_images,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_frames.append(frame_number)
                logger.error(f"帧 {frame_number} 处理失败，继续下一帧。")
        except Exception as e:
            fail_count += 1
            failed_frames.append(frame_number)
            logger.error(f"帧 {frame_number} 异常: {e}，继续下一帧。")

    # 清理
    if not args.keep_workspace:
        cleanup_mvs_workspace(local_workspace_dir)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info(f"  成功: {success_count} 帧")
    logger.info(f"  失败: {fail_count} 帧")
    if failed_frames:
        logger.info(f"  失败帧列表: {failed_frames}")
    logger.info(f"  标定结果: {ref_sparse_dir}")
    logger.info(f"  去畸变图像: {output_dir / 'undistorter_images'}")
    logger.info(f"  点云输出: {output_dir / 'points_cloud'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
