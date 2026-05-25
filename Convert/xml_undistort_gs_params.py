#!/usr/bin/env python3
"""
图像畸变矫正脚本 - 用于3DGS训练

功能:
1. 读取Agisoft Metashape导出的cameras.xml文件
2. 对图像进行去畸变处理
3. 裁切黑边，确保无黑边
4. 使Cx和Cy位于图像中心点
5. 导出适用于3DGS训练的图像和更新后的相机参数

使用方法:
python Convert/xml_undistort_gs_params.py --xml  --images_dir  --output_dir  --no_unified_intrinsics

作者: GitHub Copilot
"""

import argparse
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
import copy
import cv2
import numpy as np


def parse_agisoft_xml(xml_path: str) -> Tuple[List[Dict], List[Dict]]:
    """解析Agisoft Metashape XML文件，提取传感器标定和相机信息
    
    Returns:
        sensors: 传感器列表，包含标定信息
        cameras: 相机列表，包含sensor_id和标签
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    sensors = []
    cameras = []
    
    # 查找chunk
    chunk = root.find('.//chunk')
    if chunk is None:
        raise RuntimeError("XML中未找到chunk元素")
    
    # 解析sensors
    sensors_elem = chunk.find('sensors')
    if sensors_elem is not None:
        for sensor in sensors_elem.findall('sensor'):
            sensor_id = int(sensor.get('id', -1))
            sensor_type = sensor.get('type', 'frame')
            
            # 获取分辨率
            resolution = sensor.find('resolution')
            width = int(resolution.get('width', 0)) if resolution is not None else 0
            height = int(resolution.get('height', 0)) if resolution is not None else 0
            
            # 获取标定参数
            calibration = sensor.find('calibration')
            calib_data = {
                'f': 0.0, 'cx': 0.0, 'cy': 0.0,
                'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0,
                'p1': 0.0, 'p2': 0.0,
                'b1': 0.0, 'b2': 0.0
            }
            
            if calibration is not None:
                # 获取标定分辨率（如果与传感器不同）
                calib_res = calibration.find('resolution')
                if calib_res is not None:
                    width = int(calib_res.get('width', width))
                    height = int(calib_res.get('height', height))
                
                # 解析标定参数
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
            
            # 获取transform（如果存在）
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
    """根据Agisoft参数构建相机矩阵
    
    在Agisoft中，cx和cy是相对于图像中心的偏移量（像素单位）
    OpenCV使用的cx和cy是绝对像素坐标
    """
    cx_abs = width / 2.0 + cx
    cy_abs = height / 2.0 + cy
    
    K = np.array([
        [f, 0, cx_abs],
        [0, f, cy_abs],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def build_dist_coeffs(calib: Dict) -> np.ndarray:
    """从Agisoft标定参数构建畸变系数
    
    Agisoft使用: k1, k2, k3, k4 径向畸变
                 p1, p2 切向畸变
    OpenCV标准: [k1, k2, p1, p2, k3, k4, k5, k6]
    """
    k1 = calib.get('k1', 0.0)
    k2 = calib.get('k2', 0.0)
    k3 = calib.get('k3', 0.0)
    k4 = calib.get('k4', 0.0)
    p1 = calib.get('p1', 0.0)
    p2 = calib.get('p2', 0.0)
    
    # OpenCV 5参数模型: [k1, k2, p1, p2, k3]
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return dist


def compute_valid_roi(K: np.ndarray, dist: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """计算去畸变后的有效区域（无黑边），确保对称裁切使主点居中
    
    通过采样图像边界上的点，找到去畸变后的有效矩形区域
    关键：强制对称裁切，确保裁切后主点仍在图像中心
    
    Returns:
        (x, y, w, h): 有效区域的ROI（对称的）
    """
    # 创建边界上的点
    num_samples = 100
    
    # 上边
    top = np.array([[i, 0] for i in np.linspace(0, width - 1, num_samples)])
    # 下边
    bottom = np.array([[i, height - 1] for i in np.linspace(0, width - 1, num_samples)])
    # 左边
    left = np.array([[0, i] for i in np.linspace(0, height - 1, num_samples)])
    # 右边
    right = np.array([[width - 1, i] for i in np.linspace(0, height - 1, num_samples)])
    
    # 合并所有边界点
    border_points = np.vstack([top, bottom, left, right]).astype(np.float32)
    border_points = border_points.reshape(-1, 1, 2)
    
    # 新相机矩阵（主点居中）
    new_K = K.copy()
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    
    # 对边界点进行去畸变
    undist_points = cv2.undistortPoints(border_points, K, dist, P=new_K)
    undist_points = undist_points.reshape(-1, 2)
    
    # 找到内接矩形
    # 对于每条边，找到最内侧的点
    top_undist = undist_points[:num_samples]
    bottom_undist = undist_points[num_samples:2*num_samples]
    left_undist = undist_points[2*num_samples:3*num_samples]
    right_undist = undist_points[3*num_samples:]
    
    # 计算各边需要裁切的量
    left_crop = max(np.max(left_undist[:, 0]) - 0, 0)  # 左边界向内移动的量
    right_crop = max((width - 1) - np.min(right_undist[:, 0]), 0)  # 右边界向内移动的量
    top_crop = max(np.max(top_undist[:, 1]) - 0, 0)  # 上边界向内移动的量
    bottom_crop = max((height - 1) - np.min(bottom_undist[:, 1]), 0)  # 下边界向内移动的量
    
    # 关键：对称裁切，取左右最大值，上下最大值
    # 这确保裁切后主点仍在图像中心
    horiz_crop = max(left_crop, right_crop)
    vert_crop = max(top_crop, bottom_crop)
    
    # 转换为整数ROI（对称）
    x = int(np.ceil(horiz_crop))
    y = int(np.ceil(vert_crop))
    w = width - 2 * x  # 对称裁切，左右各裁x
    h = height - 2 * y  # 对称裁切，上下各裁y
    
    # 确保宽高为正
    w = max(w, 1)
    h = max(h, 1)
    
    return x, y, w, h


def undistort_and_crop_for_3dgs(img: np.ndarray, K: np.ndarray, dist: np.ndarray,
                                 roi: Tuple[int, int, int, int] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """去畸变并裁切以获得无黑边的图像，主点居中
    
    Args:
        img: 输入图像
        K: 原始相机矩阵
        dist: 畸变系数
        roi: 预计算的ROI，如果为None则自动计算
    
    Returns:
        undist_cropped: 去畸变并裁切后的图像
        new_K: 新的相机矩阵（主点居中）
        roi: 使用的ROI (x, y, w, h)
    """
    h, w = img.shape[:2]
    
    # 检查是否有畸变需要矫正
    has_distortion = np.any(np.abs(dist) > 1e-8)
    
    if not has_distortion:
        # 无畸变，直接返回原图
        new_K = K.copy()
        new_K[0, 2] = w / 2.0
        new_K[1, 2] = h / 2.0
        return img.copy(), new_K, (0, 0, w, h)
    
    # 创建新相机矩阵，主点居中
    new_K = K.copy()
    new_K[0, 2] = w / 2.0
    new_K[1, 2] = h / 2.0
    
    # 计算去畸变映射
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R=None, newCameraMatrix=new_K,
        size=(w, h), m1type=cv2.CV_32FC1
    )
    
    # 应用去畸变 - 使用INTER_CUBIC（双三次插值）提高质量，与COLMAP一致
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 计算或使用提供的ROI
    if roi is None:
        roi = compute_valid_roi(K, dist, w, h)
    
    x, y, crop_w, crop_h = roi
    
    # 确保ROI在图像范围内
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    crop_w = min(crop_w, w - x)
    crop_h = min(crop_h, h - y)
    
    # 裁切图像
    undist_cropped = undist[y:y+crop_h, x:x+crop_w]
    
    # 更新相机矩阵以反映裁切
    final_K = new_K.copy()
    final_K[0, 2] = crop_w / 2.0  # 新的主点x（居中）
    final_K[1, 2] = crop_h / 2.0  # 新的主点y（居中）
    
    return undist_cropped, final_K, roi


def match_image_to_camera(img_path: str, cameras: List[Dict]) -> Optional[Dict]:
    """通过标签匹配图像文件到相机"""
    img_name = Path(img_path).stem  # 不带扩展名的文件名
    
    for camera in cameras:
        if camera['label'] == img_name:
            return camera
    
    # 尝试部分匹配
    for camera in cameras:
        if img_name in camera['label'] or camera['label'] in img_name:
            return camera
    
    return None


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """将3x3旋转矩阵转换为四元数 (qw, qx, qy, qz)
    
    使用Shepperd方法，数值稳定
    """
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
    """将Agisoft的4x4变换矩阵转换为COLMAP格式
    
    Agisoft存储的是 camera-to-world 变换矩阵
    COLMAP需要的是 world-to-camera 的四元数和平移
    
    Args:
        transform: 16个浮点数的列表，按行优先存储的4x4矩阵
    
    Returns:
        quaternion: (qw, qx, qy, qz)
        translation: (tx, ty, tz)
    """
    # 重构4x4矩阵 (行优先)
    T_c2w = np.array(transform).reshape(4, 4)
    
    # 提取旋转和平移 (camera-to-world)
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    
    # 转换为 world-to-camera (COLMAP格式)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    # 转换旋转矩阵为四元数
    qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
    
    return (qw, qx, qy, qz), (t_w2c[0], t_w2c[1], t_w2c[2])


def export_colmap_cameras(output_dir: str, cameras_info: List[Dict], use_unified_intrinsics: bool = True):
    """导出COLMAP格式的相机参数文件
    
    Args:
        output_dir: 输出目录
        cameras_info: 相机信息列表
        use_unified_intrinsics: 是否使用统一的内参（推荐用于3DGS）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cameras_txt_path = os.path.join(output_dir, 'cameras.txt')
    
    if use_unified_intrinsics and cameras_info:
        # 使用第一个相机的内参作为统一内参
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
    
    print(f"COLMAP相机参数已保存到: {cameras_txt_path}")


def export_colmap_images(output_dir: str, cameras_info: List[Dict], cameras: List[Dict], 
                         use_unified_intrinsics: bool = True):
    """导出COLMAP格式的images.txt文件
    
    Args:
        output_dir: 输出目录
        cameras_info: 相机信息列表（包含label和图像名）
        cameras: 原始相机列表（包含transform）
        use_unified_intrinsics: 是否使用统一的内参
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images_txt_path = os.path.join(output_dir, 'images.txt')
    
    # 创建label到camera的映射
    label_to_camera = {cam['label']: cam for cam in cameras}
    
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of registered images: {len(cameras_info)}\n")
        
        for i, cam_info in enumerate(cameras_info, 1):
            label = cam_info['label']
            camera = label_to_camera.get(label)
            
            if camera is None or camera['transform'] is None:
                print(f"  警告: 相机 {label} 没有变换矩阵，跳过")
                continue
            
            # 转换变换矩阵为COLMAP格式
            quaternion, translation = transform_to_colmap(camera['transform'])
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation
            
            # 图像文件名
            img_name = cam_info.get('filename', f"{label}.png")
            
            # CAMERA_ID: 如果使用统一内参则为1，否则为图像序号
            camera_id = 1 if use_unified_intrinsics else i
            
            # 写入图像行
            f.write(f"{i} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} {tx:.10f} {ty:.10f} {tz:.10f} {camera_id} {img_name}\n")
            # 写入空的POINTS2D行
            f.write("\n")
    
    print(f"COLMAP图像参数已保存到: {images_txt_path}")


def export_colmap_points3d(output_dir: str):
    """导出空的COLMAP points3D.txt文件
    
    3DGS训练时会重新生成点云，所以这里创建空文件即可
    """
    os.makedirs(output_dir, exist_ok=True)
    
    points3d_txt_path = os.path.join(output_dir, 'points3D.txt')
    
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    print(f"COLMAP点云文件已保存到: {points3d_txt_path}")


def update_xml_calibration(xml_path: str, output_xml_path: str,
                           undist_info: Dict[str, Dict], new_size: Tuple[int, int]):
    """更新XML文件中的标定参数
    
    Args:
        xml_path: 原始XML路径
        output_xml_path: 输出XML路径
        undist_info: 去畸变后的内参信息 {label: {f, cx, cy, width, height}}
        new_size: 新的图像尺寸 (width, height)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    chunk = root.find('.//chunk')
    if chunk is None:
        return
    
    sensors_elem = chunk.find('sensors')
    if sensors_elem is None:
        return
    
    cameras_elem = chunk.find('cameras')
    sensor_to_label = {}
    if cameras_elem is not None:
        for camera in cameras_elem.findall('camera'):
            sensor_id = int(camera.get('sensor_id', -1))
            label = camera.get('label', '')
            if sensor_id not in sensor_to_label:
                sensor_to_label[sensor_id] = label
    
    new_width, new_height = new_size
    
    for sensor_elem in sensors_elem.findall('sensor'):
        sensor_id = int(sensor_elem.get('id', -1))
        label = sensor_to_label.get(sensor_id, '')
        
        info = undist_info.get(label)
        if info is None:
            continue
        
        # 更新分辨率
        resolution = sensor_elem.find('resolution')
        if resolution is not None:
            resolution.set('width', str(new_width))
            resolution.set('height', str(new_height))
        
        calibration = sensor_elem.find('calibration')
        if calibration is None:
            calibration = ET.SubElement(sensor_elem, 'calibration')
            calibration.set('type', 'frame')
            calibration.set('class', 'adjusted')
        
        # 更新标定参数
        def set_element(parent, tag, value):
            elem = parent.find(tag)
            if elem is None:
                elem = ET.SubElement(parent, tag)
            elem.text = str(value)
        
        # 更新标定分辨率
        calib_res = calibration.find('resolution')
        if calib_res is not None:
            calib_res.set('width', str(new_width))
            calib_res.set('height', str(new_height))
        else:
            calib_res = ET.SubElement(calibration, 'resolution')
            calib_res.set('width', str(new_width))
            calib_res.set('height', str(new_height))
        
        # 更新焦距和主点（cx, cy设为0表示居中）
        set_element(calibration, 'f', info['f'])
        set_element(calibration, 'cx', 0.0)  # 主点居中
        set_element(calibration, 'cy', 0.0)  # 主点居中
        
        # 畸变参数设为0
        set_element(calibration, 'k1', 0.0)
        set_element(calibration, 'k2', 0.0)
        set_element(calibration, 'k3', 0.0)
        set_element(calibration, 'k4', 0.0)
        set_element(calibration, 'p1', 0.0)
        set_element(calibration, 'p2', 0.0)
    
    # 写入更新后的XML
    tree.write(output_xml_path, encoding='UTF-8', xml_declaration=True)
    print(f"更新后的标定XML已保存到: {output_xml_path}")


def compute_unified_roi(sensors: List[Dict], width: int, height: int) -> Tuple[int, int, int, int]:
    """计算所有相机的统一ROI（取最保守的裁切）
    
    这确保所有图像使用相同的裁切区域，保持一致的图像尺寸
    由于compute_valid_roi已确保对称裁切，这里只需要取最大的x和y
    """
    max_x, max_y = 0, 0
    
    for sensor in sensors:
        calib = sensor['calibration']
        K = build_camera_matrix(calib['f'], calib['cx'], calib['cy'], width, height)
        dist = build_dist_coeffs(calib)
        
        # 检查是否有畸变
        if not np.any(np.abs(dist) > 1e-8):
            continue
        
        x, y, w, h = compute_valid_roi(K, dist, width, height)
        
        # 取最大的裁切量（最保守）
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # 计算最终的对称ROI
    final_w = width - 2 * max_x
    final_h = height - 2 * max_y
    
    return max_x, max_y, final_w, final_h


def main():
    parser = argparse.ArgumentParser(
        description='图像畸变矫正脚本 - 用于3DGS训练（无黑边，主点居中）')
    parser.add_argument('--xml', type=str, required=True,
                        help='Agisoft cameras.xml文件路径')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出图像目录')
    parser.add_argument('--output_xml', type=str, default=None,
                        help='输出更新后的XML文件路径（可选）')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='图像文件匹配模式（默认: *.png）')
    parser.add_argument('--unified_roi', action='store_true', default=True,
                        help='使用统一的ROI裁切所有图像（默认启用）')
    parser.add_argument('--no_unified_roi', action='store_false', dest='unified_roi',
                        help='禁用统一ROI裁切')
    parser.add_argument('--unified_intrinsics', action='store_true', default=True,
                        help='使用统一的相机内参（1个相机，默认启用）')
    parser.add_argument('--no_unified_intrinsics', action='store_false', dest='unified_intrinsics',
                        help='每张图像单独的相机内参')
    parser.add_argument('--export_colmap', action='store_true', default=True,
                        help='导出COLMAP格式的相机参数（默认启用）')
    parser.add_argument('--no_export_colmap', action='store_false', dest='export_colmap',
                        help='不导出COLMAP格式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析XML
    print(f"正在解析XML文件: {args.xml}")
    sensors, cameras = parse_agisoft_xml(args.xml)
    print(f"找到 {len(sensors)} 个传感器, {len(cameras)} 个相机")
    
    # 查找图像
    image_paths = sorted(glob.glob(os.path.join(args.images_dir, args.pattern)))
    if not image_paths:
        # 尝试其他常见格式
        for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths = sorted(glob.glob(os.path.join(args.images_dir, ext)))
            if image_paths:
                break
    
    if not image_paths:
        raise RuntimeError(f"在 {args.images_dir} 中未找到图像")
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 获取图像尺寸
    sample_img = cv2.imread(image_paths[0])
    if sample_img is None:
        raise RuntimeError(f"无法读取图像: {image_paths[0]}")
    orig_height, orig_width = sample_img.shape[:2]
    print(f"原始图像尺寸: {orig_width} x {orig_height}")
    
    # 计算统一ROI
    unified_roi = None
    if args.unified_roi:
        print("正在计算统一的裁切区域...")
        unified_roi = compute_unified_roi(sensors, orig_width, orig_height)
        x, y, w, h = unified_roi
        print(f"统一ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"输出图像尺寸: {w} x {h}")
    
    # 处理每张图像
    cameras_info = []
    undist_info = {}
    new_size = None
    
    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem
        print(f"[{i+1}/{len(image_paths)}] 处理: {img_name}")
        
        # 匹配相机
        camera = match_image_to_camera(img_path, cameras)
        if camera is None:
            print(f"  警告: 未找到匹配的相机，跳过")
            continue
        
        # 获取传感器
        sensor = get_sensor_for_camera(camera, sensors)
        if sensor is None:
            print(f"  警告: 未找到传感器，跳过")
            continue
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图像，跳过")
            continue
        
        # 构建相机矩阵和畸变系数
        calib = sensor['calibration']
        K = build_camera_matrix(calib['f'], calib['cx'], calib['cy'], orig_width, orig_height)
        dist = build_dist_coeffs(calib)
        
        # 去畸变并裁切
        undist_img, new_K, roi = undistort_and_crop_for_3dgs(img, K, dist, unified_roi)
        
        if new_size is None:
            new_size = (undist_img.shape[1], undist_img.shape[0])
        
        # 保存图像到 images/ 子目录
        images_output_dir = os.path.join(args.output_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)
        output_path = os.path.join(images_output_dir, Path(img_path).name)
        cv2.imwrite(output_path, undist_img)
        
        # 记录相机信息
        cam_info = {
            'label': camera['label'],
            'filename': Path(img_path).name,  # 图像文件名
            'width': undist_img.shape[1],
            'height': undist_img.shape[0],
            'fx': new_K[0, 0],
            'fy': new_K[1, 1],
            'cx': new_K[0, 2],
            'cy': new_K[1, 2],
            'f': calib['f']  # 原始焦距
        }
        cameras_info.append(cam_info)
        undist_info[camera['label']] = cam_info
    
    print(f"\n处理完成! 共处理 {len(cameras_info)} 张图像")
    print(f"输出目录: {args.output_dir}")
    
    if new_size:
        print(f"输出图像尺寸: {new_size[0]} x {new_size[1]}")
    
    # 导出COLMAP格式 (标准3DGS目录结构: sparse/0/)
    # use_unified_intrinsics=True: 所有图像共享1个相机内参（推荐，适用于同一相机拍摄）
    # use_unified_intrinsics=False: 每张图像单独的相机内参
    if args.export_colmap and cameras_info:
        colmap_dir = os.path.join(args.output_dir, 'sparse', '0')
        use_unified = args.unified_intrinsics
        export_colmap_cameras(colmap_dir, cameras_info, use_unified_intrinsics=use_unified)
        export_colmap_images(colmap_dir, cameras_info, cameras, use_unified_intrinsics=use_unified)
        export_colmap_points3d(colmap_dir)
    
    # 更新XML
    if args.output_xml and new_size:
        update_xml_calibration(args.xml, args.output_xml, undist_info, new_size)
    
    # 打印摘要
    print("\n=== 3DGS训练参数摘要 ===")
    if cameras_info:
        first = cameras_info[0]
        print(f"图像尺寸: {first['width']} x {first['height']}")
        print(f"焦距 (fx=fy): {first['fx']:.4f}")
        print(f"主点 (cx, cy): ({first['cx']:.4f}, {first['cy']:.4f})")
        print(f"主点位于图像中心: {'是' if abs(first['cx'] - first['width']/2) < 1 and abs(first['cy'] - first['height']/2) < 1 else '否'}")


if __name__ == '__main__':
    main()
