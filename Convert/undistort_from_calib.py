#!/usr/bin/env python3
"""
使用pycolmap对libCalib格式的calib.json进行图像去畸变

特点:
- 使用FULL_OPENCV模型包含k3参数，避免鱼眼效果
- 自动裁剪使主点居中（3DGS训练要求）
- 统一所有图像尺寸
- 输出COLMAP格式（PINHOLE模型，无畸变）

使用方法:
python undistort_from_calib.py --calib calib.json --images_dir images --output_dir undistorted
"""

import argparse
import json
import os
from pathlib import Path
import glob
from typing import Dict, Any, Tuple, List, Optional
import shutil
import numpy as np
import cv2

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
    print("警告: pycolmap未安装，请运行 pip install pycolmap")


# =============================================================================
# Calib.json 解析
# =============================================================================

def extract_intrinsics(cam_entry: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Dict[str, float]]:
    """从相机条目提取内参 (fx, fy), (cx, cy), (width, height), dist_coeffs"""
    model = cam_entry.get('model', {})
    data = model.get('ptr_wrapper', {}).get('data', {})
    params = data.get('parameters', {})
    crt = data.get('CameraModelCRT', {})
    base = crt.get('CameraModelBase', {})
    img_size = base.get('imageSize', {})
    width = int(img_size.get('width', 0))
    height = int(img_size.get('height', 0))

    f = params.get('f', {}).get('val')
    ar = params.get('ar', {}).get('val', 1.0)
    fx = float(f) if f is not None else float(params.get('fx', {}).get('val', 0.0))
    fy = float(fx * ar) if f is not None else float(params.get('fy', {}).get('val', 0.0))
    cx = float(params.get('cx', {}).get('val', 0.0))
    cy = float(params.get('cy', {}).get('val', 0.0))

    # 畸变系数
    dist = {
        'k1': float(params.get('k1', {}).get('val', 0.0)),
        'k2': float(params.get('k2', {}).get('val', 0.0)),
        'k3': float(params.get('k3', {}).get('val', 0.0)),
        'k4': float(params.get('k4', {}).get('val', 0.0)),
        'p1': float(params.get('p1', {}).get('val', 0.0)),
        'p2': float(params.get('p2', {}).get('val', 0.0)),
    }
    
    return (fx, fy), (cx, cy), (width, height), dist


def extract_extrinsics(cam_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """从相机条目提取外参（4x4变换矩阵，world-to-camera）"""
    # 优先从 cam_entry['transform'] 读取（libCalib 格式）
    transform = cam_entry.get('transform', {})
    if transform:
        rotation = transform.get('rotation', {})
        translation = transform.get('translation', {})
    else:
        # 回退到 model.ptr_wrapper.data.pose
        model = cam_entry.get('model', {})
        data = model.get('ptr_wrapper', {}).get('data', {})
        pose = data.get('pose', {})
        rotation = pose.get('rotation', {})
        translation = pose.get('translation', {})
    
    if not rotation or not translation:
        return None
    
    # 旋转格式判断
    if 'rx' in rotation:
        # Rodrigues 向量格式 (rx, ry, rz)
        rvec = np.array([rotation['rx'], rotation['ry'], rotation['rz']], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
    elif 'w' in rotation:
        # 四元数格式 (w, x, y, z)
        qw = rotation.get('w', 1.0)
        qx = rotation.get('x', 0.0)
        qy = rotation.get('y', 0.0)
        qz = rotation.get('z', 0.0)
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    elif 'data' in rotation:
        # 旋转矩阵格式
        R = np.array(rotation['data']).reshape(3, 3)
    else:
        return None
    
    # 平移
    if 'data' in translation:
        t = np.array(translation['data']).reshape(3)
    elif 'x' in translation:
        t = np.array([translation['x'], translation['y'], translation['z']])
    else:
        return None
    
    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数转旋转矩阵"""
    # 归一化
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-10:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """旋转矩阵转四元数"""
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


# =============================================================================
# COLMAP文件操作
# =============================================================================

def create_colmap_sparse(cameras_data: List[Dict], output_dir: str) -> str:
    """创建COLMAP稀疏重建文件（文本格式，带畸变参数）"""
    sparse_dir = os.path.join(output_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    
    # 写入cameras.txt（FULL_OPENCV模型，包含k3）
    cameras_txt_path = os.path.join(sparse_dir, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_data)}\n")
        
        for cam in cameras_data:
            cam_id = cam['id']
            width = cam['width']
            height = cam['height']
            fx, fy = cam['fx'], cam['fy']
            cx, cy = cam['cx'], cam['cy']
            dist = cam['dist']
            
            k1 = dist['k1']
            k2 = dist['k2']
            k3 = dist['k3']
            k4 = dist.get('k4', 0.0)
            p1 = dist['p1']
            p2 = dist['p2']
            k5, k6 = 0.0, 0.0
            
            # FULL_OPENCV模型: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            f.write(f"{cam_id} FULL_OPENCV {width} {height} "
                    f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f} "
                    f"{k1:.10f} {k2:.10f} {p1:.10f} {p2:.10f} "
                    f"{k3:.10f} {k4:.10f} {k5:.10f} {k6:.10f}\n")
    
    # 写入images.txt
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of registered images: {len(cameras_data)}\n")
        
        for cam in cameras_data:
            cam_id = cam['id']
            qw, qx, qy, qz = cam['quaternion']
            tx, ty, tz = cam['translation']
            img_name = cam['image_name']
            
            f.write(f"{cam_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} {cam_id} {img_name}\n")
            f.write("\n")
    
    # 写入空的points3D.txt
    points3d_txt_path = os.path.join(sparse_dir, 'points3D.txt')
    with open(points3d_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    return sparse_dir


def parse_cameras_txt(cameras_path: str) -> dict:
    """解析cameras.txt文件"""
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def parse_images_txt(images_path: str) -> dict:
    """解析images.txt文件"""
    images = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            
            images[img_id] = {
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'tx': tx, 'ty': ty, 'tz': tz,
                'camera_id': cam_id,
                'name': name
            }
            i += 2
        else:
            i += 1
    
    return images


# =============================================================================
# 主点居中处理
# =============================================================================

def compute_unified_crop(cameras: dict) -> Tuple[int, int, dict]:
    """计算统一的裁剪参数，确保所有相机的主点都在中心"""
    crop_info = {}
    
    for cam_id, cam in cameras.items():
        width = cam['width']
        height = cam['height']
        params = cam['params']
        
        if cam['model'] == 'PINHOLE' and len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            print(f"警告: Camera {cam_id} 模型 {cam['model']} 不支持")
            continue
        
        left = cx
        right = width - cx
        top = cy
        bottom = height - cy
        
        half_w = min(left, right)
        half_h = min(top, bottom)
        
        crop_info[cam_id] = {
            'new_width': int(2 * half_w),
            'new_height': int(2 * half_h),
            'fx': fx,
            'fy': fy,
            'old_cx': cx,
            'old_cy': cy
        }
    
    if not crop_info:
        raise RuntimeError("没有有效的相机数据")
    
    min_width = min(c['new_width'] for c in crop_info.values())
    min_height = min(c['new_height'] for c in crop_info.values())
    
    unified_width = min_width - (min_width % 2)
    unified_height = min_height - (min_height % 2)
    
    for cam_id in crop_info:
        cam = cameras[cam_id]
        cx, cy = cam['params'][2], cam['params'][3]
        
        half_w = unified_width / 2
        half_h = unified_height / 2
        
        crop_info[cam_id]['unified_width'] = unified_width
        crop_info[cam_id]['unified_height'] = unified_height
        crop_info[cam_id]['crop_x'] = int(cx - half_w)
        crop_info[cam_id]['crop_y'] = int(cy - half_h)
    
    return unified_width, unified_height, crop_info


def center_principal_point(undistorted_dir: str, output_dir: str) -> Tuple[int, int]:
    """裁剪去畸变后的图像使主点居中"""
    input_sparse = os.path.join(undistorted_dir, 'sparse')
    input_images = os.path.join(undistorted_dir, 'images')
    
    # 如果是二进制格式，先转换
    if os.path.exists(os.path.join(input_sparse, 'cameras.bin')):
        print("  转换二进制格式到文本...")
        rec = pycolmap.Reconstruction(input_sparse)
        sparse_txt = os.path.join(undistorted_dir, 'sparse_txt')
        os.makedirs(sparse_txt, exist_ok=True)
        rec.write_text(sparse_txt)
        input_sparse = sparse_txt
    
    cameras = parse_cameras_txt(os.path.join(input_sparse, 'cameras.txt'))
    images = parse_images_txt(os.path.join(input_sparse, 'images.txt'))
    
    unified_width, unified_height, crop_info = compute_unified_crop(cameras)
    print(f"  统一输出尺寸: {unified_width} x {unified_height}")
    print(f"  主点位置: ({unified_width/2}, {unified_height/2}) [完全居中]")
    
    output_images = os.path.join(output_dir, 'images')
    output_sparse = os.path.join(output_dir, 'sparse')
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_sparse, exist_ok=True)
    
    print("  裁剪图像...")
    for img_id, img_data in images.items():
        cam_id = img_data['camera_id']
        img_name = img_data['name']
        
        if cam_id not in crop_info:
            continue
        
        crop = crop_info[cam_id]
        img_path = os.path.join(input_images, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        crop_x = max(0, min(crop['crop_x'], w - crop['unified_width']))
        crop_y = max(0, min(crop['crop_y'], h - crop['unified_height']))
        crop_w = crop['unified_width']
        crop_h = crop['unified_height']
        
        cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        cv2.imwrite(os.path.join(output_images, img_name), cropped)
    
    # 写入cameras.txt（主点在中心）
    with open(os.path.join(output_sparse, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        
        for cam_id, cam in cameras.items():
            if cam_id not in crop_info:
                continue
            fx = crop_info[cam_id]['fx']
            fy = crop_info[cam_id]['fy']
            cx = unified_width / 2.0
            cy = unified_height / 2.0
            f.write(f"{cam_id} PINHOLE {unified_width} {unified_height} "
                    f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")
    
    # 复制images.txt
    shutil.copy(os.path.join(input_sparse, 'images.txt'),
                os.path.join(output_sparse, 'images.txt'))
    
    # 创建空的points3D.txt
    with open(os.path.join(output_sparse, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    return unified_width, unified_height


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='使用pycolmap对calib.json进行图像去畸变 - 用于3DGS训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python undistort_from_calib.py --calib calib.json --images_dir images --output_dir output
        """)
    parser.add_argument('--calib', type=str, required=True,
                        help='calib.json文件路径')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='原始图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='图像文件匹配模式 (默认: *.png)')
    
    args = parser.parse_args()
    
    if not HAS_PYCOLMAP:
        print("错误: pycolmap未安装，请运行 pip install pycolmap")
        return 1
    
    print("="*60)
    print("Calib.json图像去畸变工具 (pycolmap)")
    print("="*60)
    
    # 创建临时目录
    temp_dir = os.path.join(args.output_dir, '_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 步骤1: 解析calib.json
    print(f"\n[1/4] 解析calib.json: {args.calib}")
    with open(args.calib, 'r') as f:
        calib = json.load(f)
    
    cams = calib.get('Calibration', {}).get('cameras', [])
    if not cams:
        raise RuntimeError('calib.json中未找到相机')
    
    print(f"  找到 {len(cams)} 个相机")
    
    # 步骤2: 查找图像并匹配相机
    print(f"\n[2/4] 准备图像...")
    image_paths = sorted(glob.glob(os.path.join(args.images_dir, args.pattern)))
    if not image_paths:
        for ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths = sorted(glob.glob(os.path.join(args.images_dir, ext)))
            if image_paths:
                break
    
    if not image_paths:
        raise RuntimeError(f"在 {args.images_dir} 中未找到图像")
    
    print(f"  找到 {len(image_paths)} 张图像")
    
    # 准备图像目录
    images_input_dir = os.path.join(temp_dir, 'images_input')
    os.makedirs(images_input_dir, exist_ok=True)
    
    # 图像和相机一一对应
    per_cam = len(image_paths) == len(cams)
    
    cameras_data = []
    for i, img_path in enumerate(image_paths):
        cam_idx = i if per_cam else 0
        cam_entry = cams[cam_idx]
        
        # 提取内参
        (fx, fy), (cx, cy), (w0, h0), dist = extract_intrinsics(cam_entry)
        
        # 读取图像获取实际尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取 {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # 如果图像尺寸与标定尺寸不同，缩放内参
        if w0 > 0 and h0 > 0 and (w != w0 or h != h0):
            sx, sy = w / w0, h / h0
            fx, fy = fx * sx, fy * sy
            cx, cy = cx * sx, cy * sy
        
        # 提取外参
        T_w2c = extract_extrinsics(cam_entry)
        if T_w2c is None:
            # 如果没有外参，使用单位矩阵
            T_w2c = np.eye(4)
        
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
        
        # 复制图像（移除空格）
        new_name = Path(img_path).name.replace(' ', '')
        shutil.copy2(img_path, os.path.join(images_input_dir, new_name))
        
        cameras_data.append({
            'id': i + 1,
            'width': w,
            'height': h,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'dist': dist,
            'quaternion': (qw, qx, qy, qz),
            'translation': (t[0], t[1], t[2]),
            'image_name': new_name
        })
    
    print(f"  已处理 {len(cameras_data)} 张图像")
    
    # 创建COLMAP稀疏模型
    sparse_dir = create_colmap_sparse(cameras_data, temp_dir)
    print(f"  已创建COLMAP稀疏模型")
    
    # 步骤3: pycolmap去畸变
    print(f"\n[3/4] 使用pycolmap去畸变...")
    undistorted_dir = os.path.join(temp_dir, 'undistorted')
    
    pycolmap.undistort_images(
        output_path=undistorted_dir,
        input_path=sparse_dir,
        image_path=images_input_dir,
        output_type="COLMAP"
    )
    print(f"  去畸变完成")
    
    # 步骤4: 裁剪使主点居中
    print(f"\n[4/4] 裁剪图像使主点居中...")
    unified_width, unified_height = center_principal_point(undistorted_dir, args.output_dir)
    
    # 清理临时目录
    print(f"\n清理临时文件...")
    shutil.rmtree(temp_dir)
    
    # 完成
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    print(f"  - 图像: {os.path.join(args.output_dir, 'images')}")
    print(f"  - cameras.txt: {os.path.join(args.output_dir, 'sparse', 'cameras.txt')}")
    print(f"  - images.txt: {os.path.join(args.output_dir, 'sparse', 'images.txt')}")
    print(f"  - points3D.txt: {os.path.join(args.output_dir, 'sparse', 'points3D.txt')}")
    print(f"\n图像尺寸: {unified_width} x {unified_height}")
    print(f"主点位置: ({unified_width/2}, {unified_height/2}) [完全居中]")
    print(f"相机模型: PINHOLE (无畸变)")
    print("\n✓ 可直接用于3DGS训练")
    
    return 0


if __name__ == '__main__':
    exit(main())
