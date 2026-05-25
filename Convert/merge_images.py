import os
import shutil
from pathlib import Path


def reorganize_camera_to_frame(source_dir, output_dir):
    """
    将相机为单位的目录结构重组为帧为单位的目录结构。

    输入结构:
        source_dir/
            camera_A/  (48个相机文件夹，按名称排序决定相机编号)
                img0.png
                img1.png
                ...
            camera_B/
                img0.png
                ...

    输出结构:
        output_dir/
            1/          (帧编号，按相机内文件排序决定)
                1.png   (相机1在该帧的图片)
                2.png   (相机2在该帧的图片)
                ...
                48.png
            2/
                1.png
                ...

    Args:
        source_dir: 包含相机子文件夹的源目录
        output_dir: 输出目录
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return

    # 获取所有相机文件夹，按名称排序，序号即相机编号
    cam_folders = sorted([f for f in source_path.iterdir() if f.is_dir()])
    if not cam_folders:
        print("未找到任何相机子文件夹")
        return

    print(f"找到 {len(cam_folders)} 个相机文件夹")

    # 每个相机文件夹下的图片列表（按名称排序）
    cam_frames: list[list[Path]] = []
    for cam_folder in cam_folders:
        files = sorted([f for f in cam_folder.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                        and not f.name.startswith('.')])
        cam_frames.append(files)

    # 检查帧数是否一致
    frame_counts = [len(f) for f in cam_frames]
    if len(set(frame_counts)) > 1:
        print("警告: 各相机文件夹的图片数量不一致:")
        for cam_folder, cnt in zip(cam_folders, frame_counts):
            print(f"  {cam_folder.name}: {cnt} 张")
    num_frames = max(frame_counts) if frame_counts else 0
    print(f"帧数: {num_frames}，相机数: {len(cam_folders)}")

    total_copied = 0
    total_skipped = 0

    for frame_idx in range(num_frames):
        frame_dir = Path(output_dir) / str(frame_idx + 1)
        frame_dir.mkdir(parents=True, exist_ok=True)

        for cam_idx, files in enumerate(cam_frames):
            if frame_idx >= len(files):
                print(f"  警告: 相机 {cam_idx + 1} 没有第 {frame_idx + 1} 帧，跳过")
                total_skipped += 1
                continue

            src_file = files[frame_idx]
            suffix = src_file.suffix.lower()
            dest_file = frame_dir / f"{cam_idx + 1}{suffix}"

            try:
                shutil.copy2(src_file, dest_file)
                total_copied += 1
            except Exception as e:
                print(f"  复制失败: {src_file} -> {dest_file}: {e}")
                total_skipped += 1

        print(f"  帧 {frame_idx + 1}/{num_frames} 完成")

    # 清理输出目录中所有以 '.' 开头的隐藏文件
    hidden_deleted = 0
    for hidden in Path(output_dir).rglob('*'):
        if hidden.is_file() and hidden.name.startswith('.'):
            try:
                hidden.unlink()
                hidden_deleted += 1
            except Exception as e:
                print(f"  删除隐藏文件失败: {hidden} - {e}")

    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"成功复制: {total_copied} 个文件")
    print(f"跳过/失败: {total_skipped} 个文件")
    print(f"清理隐藏文件: {hidden_deleted} 个")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    source_directory = r"images"
    output_directory = r"frames1"

    reorganize_camera_to_frame(source_directory, output_directory)
