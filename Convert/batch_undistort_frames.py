#!/usr/bin/env python3
"""
批量图像畸变矫正脚本 - 用于多帧数据的3DGS训练

功能:
1. 读取Agisoft Metashape导出的cameras.xml文件
2. 批量对images_group下每个帧文件夹中的图像进行去畸变处理
3. 裁切黑边，确保无黑边
4. 使Cx和Cy位于图像中心点
5. 导出适用于3DGS训练的图像和更新后的相机参数

目录结构:
  images_group/
    1/         # 第1帧
      1.png    # 相机1拍摄
      2.png    # 相机2拍摄
      ...
    2/         # 第2帧
      1.png
      2.png
      ...

使用方法:
python Convert/batch_undistort_frames.py --xml cameras.xml --images_group images_group --output_dir output

作者: GitHub Copilot
"""

import argparse
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def parse_agisoft_xml(xml_path: str) -> Tuple[List[Dict], List[Dict]]:
    """解析Agisoft Metashape XML文件，提取传感器标定和相机信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    sensors = []
    cameras = []
    
    chunk = root.find('.//chunk')
    if chunk is None:
        raise RuntimeError("XML中未找到chunk元素")
    
    # 解析sensors
    sensors_elem = chunk.find('sensors')
    if sensors_elem is not None:
        for sensor in sensors_elem.findall('sensor'):
            sensor_id = int(sensor.get('id', -1))
            sensor_type = sensor.get('type', 'frame')
            
            resolution = sensor.find('resolution')
            width = int(resolution.get('width', 0)) if resolution is not None else 0
            height = int(resolution.get('height', 0)) if resolution is not None else 0
            
            calibration = sensor.find('calibration')
            calib_data = {
                'f': 0.0, 'cx': 0.0, 'cy': 0.0,
                'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0,
                'p1': 0.0, 'p2': 0.0,
                'b1': 0.0, 'b2': 0.0
            }
            
            if calibration is not None:
                calib_res = calibration.find('resolution')
                if calib_res is not None:
                    width = int(calib_res.get('width', width))
                    height = int(calib_res.get('height', height))
                
                for param in ['f', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2', 'b1', 'b2']:
                    elem = calibration.find(param)
                    if elem is not None and elem.text:
                        calib_data[param] = float(elem.text)
            
            sensors.append({
                'id': sensor_id,
                'type': sensor_type,
                'width': width,
                'height': height,
                'calibration': calib_data
            })
    
    # 解析cameras
    cameras_elem = chunk.find('cameras')
    if cameras_elem is not None:
        for camera in cameras_elem.findall('camera'):
            camera_id = int(camera.get('id', -1))
            sensor_id = int(camera.get('sensor_id', -1))
            label = camera.get('label', '')
            
            transform = None
            transform_elem = camera.find('transform')
            if transform_elem is not None and transform_elem.text:
                transform = [float(x) for x in transform_elem.text.split()]
            
            cameras.append({
                'id': camera_id,
                'sensor_id': sensor_id,
                'label': label,
                'transform': transform
            })
    
    return sensors, cameras


def get_sensor_for_camera(camera: Dict, sensors: List[Dict]) -> Optional[Dict]:
    """获取与相机关联的传感器"""
    sensor_id = camera['sensor_id']
    for sensor in sensors:
        if sensor['id'] == sensor_id:
            return sensor
    return None


def build_camera_matrix(f: float, cx: float, cy: float, width: int, height: int) -> np.ndarray:
    """根据Agisoft参数构建相机矩阵"""
    cx_abs = width / 2.0 + cx
    cy_abs = height / 2.0 + cy
    
    K = np.array([
        [f, 0, cx_abs],
        [0, f, cy_abs],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def build_dist_coeffs(calib: Dict) -> np.ndarray:
    """从Agisoft标定参数构建畸变系数"""
    k1 = calib.get('k1', 0.0)
    k2 = calib.get('k2', 0.0)
    k3 = calib.get('k3', 0.0)
    p1 = calib.get('p1', 0.0)
    p2 = calib.get('p2', 0.0)
    
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return dist


def compute_valid_roi(K: np.ndarray, dist: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """计算去畸变后的有效区域（无黑边），确保对称裁切使主点居中"""
    num_samples = 100
    
    top = np.array([[i, 0] for i in np.linspace(0, width - 1, num_samples)])
    bottom = np.array([[i, height - 1] for i in np.linspace(0, width - 1, num_samples)])
    left = np.array([[0, i] for i in np.linspace(0, height - 1, num_samples)])
    right = np.array([[width - 1, i] for i in np.linspace(0, height - 1, num_samples)])
    
    border_points = np.vstack([top, bottom, left, right]).astype(np.float32)
    border_points = border_points.reshape(-1, 1, 2)
    
    new_K = K.copy()
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    
    undist_points = cv2.undistortPoints(border_points, K, dist, P=new_K)
    undist_points = undist_points.reshape(-1, 2)
    
    top_undist = undist_points[:num_samples]
    bottom_undist = undist_points[num_samples:2*num_samples]
    left_undist = undist_points[2*num_samples:3*num_samples]
    right_undist = undist_points[3*num_samples:]
    
    left_crop = max(np.max(left_undist[:, 0]) - 0, 0)
    right_crop = max((width - 1) - np.min(right_undist[:, 0]), 0)
    top_crop = max(np.max(top_undist[:, 1]) - 0, 0)
    bottom_crop = max((height - 1) - np.min(bottom_undist[:, 1]), 0)
    
    horiz_crop = max(left_crop, right_crop)
    vert_crop = max(top_crop, bottom_crop)
    
    x = int(np.ceil(horiz_crop))
    y = int(np.ceil(vert_crop))
    w = width - 2 * x
    h = height - 2 * y
    
    w = max(w, 1)
    h = max(h, 1)
    
    return x, y, w, h


def compute_unified_roi(sensors: List[Dict], width: int, height: int) -> Tuple[int, int, int, int]:
    """计算所有相机的统一ROI（取最保守的裁切）"""
    max_x, max_y = 0, 0
    
    for sensor in sensors:
        calib = sensor['calibration']
        K = build_camera_matrix(calib['f'], calib['cx'], calib['cy'], width, height)
        dist = build_dist_coeffs(calib)
        
        if not np.any(np.abs(dist) > 1e-8):
            continue
        
        x, y, w, h = compute_valid_roi(K, dist, width, height)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    final_w = width - 2 * max_x
    final_h = height - 2 * max_y
    
    return max_x, max_y, final_w, final_h


def undistort_and_crop(img: np.ndarray, K: np.ndarray, dist: np.ndarray,
                       roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """去畸变并裁切以获得无黑边的图像，主点居中"""
    h, w = img.shape[:2]
    
    has_distortion = np.any(np.abs(dist) > 1e-8)
    
    if not has_distortion:
        new_K = K.copy()
        new_K[0, 2] = w / 2.0
        new_K[1, 2] = h / 2.0
        x, y, crop_w, crop_h = roi
        cropped = img[y:y+crop_h, x:x+crop_w]
        final_K = new_K.copy()
        final_K[0, 2] = crop_w / 2.0
        final_K[1, 2] = crop_h / 2.0
        return cropped, final_K
    
    new_K = K.copy()
    new_K[0, 2] = w / 2.0
    new_K[1, 2] = h / 2.0
    
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R=None, newCameraMatrix=new_K,
        size=(w, h), m1type=cv2.CV_32FC1
    )
    
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    x, y, crop_w, crop_h = roi
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    crop_w = min(crop_w, w - x)
    crop_h = min(crop_h, h - y)
    
    undist_cropped = undist[y:y+crop_h, x:x+crop_w]
    
    final_K = new_K.copy()
    final_K[0, 2] = crop_w / 2.0
    final_K[1, 2] = crop_h / 2.0
    
    return undist_cropped, final_K


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """将3x3旋转矩阵转换为四元数 (qw, qx, qy, qz)"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return qw, qx, qy, qz


def transform_to_colmap(transform: List[float]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float]]:
    """将Agisoft的4x4变换矩阵转换为COLMAP格式"""
    T_c2w = np.array(transform).reshape(4, 4)
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
    
    return (qw, qx, qy, qz), (t_w2c[0], t_w2c[1], t_w2c[2])


def export_colmap_files(output_dir: str, cameras_info: List[Dict], cameras: List[Dict],
                        use_unified_intrinsics: bool = True):
    """导出COLMAP格式的相机参数文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # cameras.txt
    cameras_txt_path = os.path.join(output_dir, 'cameras.txt')
    if use_unified_intrinsics and cameras_info:
        first = cameras_info[0]
        with open(cameras_txt_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: 1\n")
            f.write(f"1 PINHOLE {first['width']} {first['height']} {first['fx']:.10f} {first['fy']:.10f} {first['cx']:.10f} {first['cy']:.10f}\n")
    else:
        with open(cameras_txt_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cameras_info)}\n")
            for i, cam in enumerate(cameras_info, 1):
                f.write(f"{i} PINHOLE {cam['width']} {cam['height']} {cam['fx']:.10f} {cam['fy']:.10f} {cam['cx']:.10f} {cam['cy']:.10f}\n")
    
    # images.txt
    label_to_camera = {cam['label']: cam for cam in cameras}
    images_txt_path = os.path.join(output_dir, 'images.txt')
    
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of registered images: {len(cameras_info)}\n")
        
        for i, cam_info in enumerate(cameras_info, 1):
            label = cam_info['label']
            camera = label_to_camera.get(label)
            
            if camera is None or camera['transform'] is None:
                continue
            
            quaternion, translation = transform_to_colmap(camera['transform'])
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation
            
            img_name = cam_info.get('filename', f"{label}.png")
            camera_id = 1 if use_unified_intrinsics else i
            
            f.write(f"{i} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} {tx:.10f} {ty:.10f} {tz:.10f} {camera_id} {img_name}\n")
            f.write("\n")
    
    # points3D.txt
    points3d_txt_path = os.path.join(output_dir, 'points3D.txt')
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


def process_single_frame(frame_dir: str, output_frame_dir: str, 
                         sensors: List[Dict], cameras: List[Dict],
                         unified_roi: Tuple[int, int, int, int],
                         pattern: str = '*.png') -> Tuple[str, int, int, List[Dict]]:
    """处理单个帧文件夹
    
    Returns:
        (frame_name, processed_count, skipped_count, cameras_info)
    """
    frame_name = os.path.basename(frame_dir)
    
    # 查找图像
    image_paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if not image_paths:
        for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths = sorted(glob.glob(os.path.join(frame_dir, ext)))
            if image_paths:
                break
    
    if not image_paths:
        return frame_name, 0, 0, []
    
    # 创建输出目录：标准3DGS结构 output_frame_dir/images/
    images_output_dir = os.path.join(output_frame_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    cameras_info = []
    
    # 创建label到camera的映射
    label_to_camera = {cam['label']: cam for cam in cameras}
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        
        # 查找对应的相机（通过label匹配）
        camera = label_to_camera.get(img_name)
        if camera is None:
            # 尝试部分匹配
            for cam in cameras:
                if img_name in cam['label'] or cam['label'] in img_name:
                    camera = cam
                    break
        
        if camera is None:
            skipped += 1
            continue
        
        # 获取传感器
        sensor = get_sensor_for_camera(camera, sensors)
        if sensor is None:
            skipped += 1
            continue
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        
        # 构建相机矩阵和畸变系数
        calib = sensor['calibration']
        orig_height, orig_width = img.shape[:2]
        K = build_camera_matrix(calib['f'], calib['cx'], calib['cy'], orig_width, orig_height)
        dist = build_dist_coeffs(calib)
        
        # 去畸变并裁切
        undist_img, new_K = undistort_and_crop(img, K, dist, unified_roi)
        
        # 保存图像到 images/ 子目录（标准3DGS结构）
        output_path = os.path.join(images_output_dir, Path(img_path).name)
        cv2.imwrite(output_path, undist_img)
        
        # 记录相机信息（只在第一帧时需要）
        cam_info = {
            'label': camera['label'],
            'filename': Path(img_path).name,
            'width': undist_img.shape[1],
            'height': undist_img.shape[0],
            'fx': new_K[0, 0],
            'fy': new_K[1, 1],
            'cx': new_K[0, 2],
            'cy': new_K[1, 2],
            'f': calib['f']
        }
        cameras_info.append(cam_info)
        processed += 1
    
    return frame_name, processed, skipped, cameras_info


def main():
    parser = argparse.ArgumentParser(
        description='批量图像畸变矫正脚本 - 处理多帧数据')
    parser.add_argument('--xml', type=str, required=True,
                        help='Agisoft cameras.xml文件路径')
    parser.add_argument('--images_group', type=str, required=True,
                        help='图像组目录（包含多个帧文件夹）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='图像文件匹配模式（默认: *.png）')
    parser.add_argument('--unified_intrinsics', action='store_true', default=False,
                        help='使用统一的相机内参（默认禁用，每个相机单独内参）')
    parser.add_argument('--no_unified_intrinsics', action='store_false', dest='unified_intrinsics',
                        help='每张图像单独的相机内参（默认）')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行处理的线程数（默认: 8）')
    
    args = parser.parse_args()
    
    # 解析XML
    print(f"正在解析XML文件: {args.xml}")
    sensors, cameras = parse_agisoft_xml(args.xml)
    print(f"找到 {len(sensors)} 个传感器, {len(cameras)} 个相机")
    
    # 获取帧文件夹列表
    images_group_path = Path(args.images_group)
    frame_dirs = sorted([d for d in images_group_path.iterdir() if d.is_dir()],
                        key=lambda x: int(x.name) if x.name.isdigit() else x.name)
    
    if not frame_dirs:
        raise RuntimeError(f"在 {args.images_group} 中未找到帧文件夹")
    
    print(f"找到 {len(frame_dirs)} 个帧文件夹")
    
    # 获取第一帧的第一张图像来确定尺寸
    sample_frame = frame_dirs[0]
    sample_images = sorted(glob.glob(os.path.join(str(sample_frame), args.pattern)))
    if not sample_images:
        for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            sample_images = sorted(glob.glob(os.path.join(str(sample_frame), ext)))
            if sample_images:
                break
    
    if not sample_images:
        raise RuntimeError(f"在 {sample_frame} 中未找到图像")
    
    sample_img = cv2.imread(sample_images[0])
    if sample_img is None:
        raise RuntimeError(f"无法读取图像: {sample_images[0]}")
    
    orig_height, orig_width = sample_img.shape[:2]
    print(f"原始图像尺寸: {orig_width} x {orig_height}")
    
    # 计算统一ROI
    print("正在计算统一的裁切区域...")
    unified_roi = compute_unified_roi(sensors, orig_width, orig_height)
    x, y, w, h = unified_roi
    print(f"统一ROI: x={x}, y={y}, w={w}, h={h}")
    print(f"输出图像尺寸: {w} x {h}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个帧
    print(f"\n开始处理 {len(frame_dirs)} 个帧...")
    print(f"使用 {args.workers} 个线程并行处理\n")
    
    total_processed = 0
    total_skipped = 0
    first_frame_cameras_info = None  # 保存第一帧的相机信息用于生成COLMAP文件
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for frame_dir in frame_dirs:
            frame_name = frame_dir.name
            output_frame_dir = os.path.join(args.output_dir, frame_name)
            
            future = executor.submit(
                process_single_frame,
                str(frame_dir),
                output_frame_dir,
                sensors,
                cameras,
                unified_roi,
                args.pattern
            )
            futures[future] = frame_name
        
        # 收集结果（按帧名排序以确保获取真正的第一帧）
        results = []
        for future in as_completed(futures):
            frame_name = futures[future]
            try:
                name, processed, skipped, cameras_info = future.result()
                total_processed += processed
                total_skipped += skipped
                results.append((name, cameras_info))
                print(f"  [{name}] 处理: {processed}, 跳过: {skipped}")
            except Exception as e:
                print(f"  [{frame_name}] 错误: {e}")
        
        # 按帧名排序，获取第一帧的相机信息
        results.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        for name, cameras_info in results:
            if cameras_info:
                first_frame_cameras_info = cameras_info
                break
    
    # 只生成一次COLMAP格式文件（在输出根目录的sparse/0/下，所有帧共享）
    if first_frame_cameras_info:
        colmap_dir = os.path.join(args.output_dir, 'sparse', '0')
        print(f"\n正在生成COLMAP格式文件...")
        export_colmap_files(colmap_dir, first_frame_cameras_info, cameras, args.unified_intrinsics)
        print(f"COLMAP文件已保存到: {colmap_dir}")
    
    # 打印摘要
    print(f"\n{'='*50}")
    print(f"批量处理完成!")
    print(f"{'='*50}")
    print(f"总帧数: {len(frame_dirs)}")
    print(f"总处理图像: {total_processed}")
    print(f"总跳过图像: {total_skipped}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出图像尺寸: {w} x {h}")
    print(f"帧图像目录: {{frame}}/images/")
    print(f"COLMAP参数: {args.output_dir}/sparse/0/ (所有帧共享)")
    print(f"主点居中: 是")


if __name__ == '__main__':
    main()
