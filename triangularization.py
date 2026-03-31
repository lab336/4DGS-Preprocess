"""
多相机三维点云重建 Pipeline
从多相机去畸变图像 + 已知相机内外参，使用 RoMa v2 生成三维点云。

用法:
    python main.py
    python main.py --frames 1,2,3 --roma_setting fast
    python main.py --num_samples 8000 --neighbor_range 5
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from romav2 import RoMaV2


# ======================== 配置 ========================

@dataclass
class Config:
    """Pipeline 可调参数。"""
    data_dir: str = "images_processed"       # 数据根目录
    output_dir: str = "output"               # 输出目录
    num_samples: int = 5000                  # 每对图像采样匹配数
    target_points: int = 50000               # 最终目标点数
    min_confidence: float = 0.1              # RoMa overlap 置信度过滤阈值
    ransac_threshold: float = 1.0            # RANSAC 阈值 (像素，严格提高可靠性)
    reproj_threshold: float = 2.0            # 逐对重投影误差阈值 (像素，严格)
    epipolar_threshold: float = 2.0          # 极线距离阈值 (像素，用已知位姿计算)
    max_depth: float = 100.0                 # 三角化最大深度
    min_depth: float = 0.05                  # 三角化最小深度
    min_angle: float = 1.0                   # 最小三角化角度 (度)，降低以保留花瓣
    min_observations: int = 1                # 点至少被多少个视图对独立三角化 (1=单对也接受)
    min_visible_views: int = 2               # 多视角可见性：至少被 2 个相机看到
    neighbor_range: int = 3                  # 每个相机与前后 N 个邻居配对
    sor_k: int = 10                          # 统计离群值去除 - 近邻数
    sor_std: float = 3.0                     # 统计离群值去除 - 标准差倍数
    roma_setting: str = "fast"               # RoMa v2 模式: turbo/fast/base/precise
    compile: bool = False                    # torch.compile (Windows 须关闭)


# ======================== COLMAP 文件解析 ========================

def parse_cameras(cameras_path: Path) -> dict:
    """解析 COLMAP cameras.txt，返回 {camera_id: {K, width, height}}。"""
    cameras = {}
    with open(cameras_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, parts[4:8])
            else:
                raise ValueError(f"不支持的相机模型: {model}")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            cameras[cam_id] = {"K": K, "width": width, "height": height}
    return cameras


def parse_images(images_path: Path) -> dict:
    """解析 COLMAP images.txt，返回 {image_name: {R, t, camera_id}}。

    R, t 满足: x_cam = R @ x_world + t (world-to-camera)。
    """
    images = {}
    with open(images_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        # COLMAP 四元数 -> 旋转矩阵 (scipy 用 [x,y,z,w] 格式)
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz], dtype=np.float64)
        images[name] = {"R": R, "t": t, "camera_id": camera_id}
        i += 2  # 跳过 POINTS2D 行
    return images


def scale_intrinsics(K: np.ndarray, colmap_w: int, colmap_h: int,
                     actual_w: int, actual_h: int) -> np.ndarray:
    """若实际图像分辨率与 COLMAP 不同，缩放内参矩阵。"""
    if colmap_w == actual_w and colmap_h == actual_h:
        return K
    sx = actual_w / colmap_w
    sy = actual_h / colmap_h
    K_scaled = K.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy
    return K_scaled


# ======================== 三角化与过滤 ========================

def compute_fundamental_from_poses(K1, R1, t1, K2, R2, t2):
    """从已知相机位姿直接计算 Fundamental Matrix。

    这比 RANSAC 估计 E 矩阵更可靠，因为：
    - 不会被背景主导，花瓣等小物体不会被当作 outlier
    - 所有深度的点都满足同一个极线约束
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]], dtype=np.float64)
    E = tx @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def epipolar_filter(pts1, pts2, F, threshold):
    """用已知 F 矩阵做极线距离过滤。

    Sampson 距离：综合考虑两个方向的极线距离，比单侧更鲁棒。
    优势：不会像 RANSAC 那样被背景主导，花瓣等小物体不会被误删。
    """
    n = len(pts1)
    p1h = np.hstack([pts1, np.ones((n, 1))])  # Nx3
    p2h = np.hstack([pts2, np.ones((n, 1))])
    Fp1 = (F @ p1h.T).T       # Nx3: 极线 in image2
    Ftp2 = (F.T @ p2h.T).T    # Nx3: 极线 in image1
    # p2^T F p1
    pfp = np.sum(p2h * Fp1, axis=1)
    # Sampson distance
    denom = Fp1[:, 0]**2 + Fp1[:, 1]**2 + Ftp2[:, 0]**2 + Ftp2[:, 1]**2
    sampson = pfp**2 / (denom + 1e-12)
    return sampson < threshold**2


def triangulate_points(K1, R1, t1, K2, R2, t2, pts1, pts2):
    """从两个视角三角化三维点。"""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3:]).T
    return pts3d


def compute_triangulation_angles(R1, t1, R2, t2, pts3d):
    """计算每个点的三角化角度（两相机射线夹角，度）。角度越大精度越高。"""
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    ray1 = pts3d - C1.reshape(1, 3)
    ray2 = pts3d - C2.reshape(1, 3)
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    cos_a = np.sum(ray1 * ray2, axis=1) / (n1 * n2 + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def filter_by_depth(R1, t1, R2, t2, pts3d, min_d, max_d):
    """过滤深度不合理的点。"""
    d1 = (R1 @ pts3d.T + t1.reshape(3, 1))[2]
    d2 = (R2 @ pts3d.T + t2.reshape(3, 1))[2]
    return (d1 > min_d) & (d1 < max_d) & (d2 > min_d) & (d2 < max_d)


def filter_by_reprojection(K1, R1, t1, K2, R2, t2, pts1, pts2, pts3d, thr):
    """通过重投影误差过滤不可靠的三维点。"""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
    proj1 = P1 @ h; proj1 = (proj1[:2] / proj1[2:]).T
    proj2 = P2 @ h; proj2 = (proj2[:2] / proj2[2:]).T
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    return (err1 < thr) & (err2 < thr)


# ======================== 体素合并与多视角验证 ========================

def merge_and_count_observations(points, colors):
    """体素网格合并空间相近点。

    自动根据场景包围盒计算体素大小。同一体素内的点取均值位置、
    均值颜色，并统计观测次数（被多少次独立三角化命中）。
    观测次数高的点意味着被多个相机对一致地重建出来，可靠性高。
    """
    if len(points) == 0:
        return points, colors, np.array([], dtype=int)
    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    voxel_size = max(bbox_diag * 0.003, 1e-6)
    origin = points.min(axis=0)
    vox_idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    vox_idx -= vox_idx.min(axis=0)
    dims = vox_idx.max(axis=0) + 1
    keys = (vox_idx[:, 0] * (dims[1] * dims[2])
            + vox_idx[:, 1] * dims[2]
            + vox_idx[:, 2])
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    n_vox = len(unique_keys)
    # 向量化聚合 (np.bincount 比循环快几十倍)
    mx = np.bincount(inverse, weights=points[:, 0], minlength=n_vox) / counts
    my = np.bincount(inverse, weights=points[:, 1], minlength=n_vox) / counts
    mz = np.bincount(inverse, weights=points[:, 2], minlength=n_vox) / counts
    merged_pts = np.column_stack([mx, my, mz])
    mr = np.bincount(inverse, weights=colors[:, 0].astype(float), minlength=n_vox) / counts
    mg = np.bincount(inverse, weights=colors[:, 1].astype(float), minlength=n_vox) / counts
    mb = np.bincount(inverse, weights=colors[:, 2].astype(float), minlength=n_vox) / counts
    merged_clr = np.column_stack([mr, mg, mb]).astype(np.uint8)
    return merged_pts, merged_clr, counts


def multiview_visibility_filter(pts3d, valid_images, K_cache, images_info,
                                actual_w, actual_h, min_views):
    """保留能被至少 min_views 个相机看到（正深度 + 画面内）的点。"""
    n = len(pts3d)
    if n == 0:
        return np.ones(0, dtype=bool)
    visible = np.zeros(n, dtype=int)
    h = np.hstack([pts3d, np.ones((n, 1))]).T  # 4xN
    for name in valid_images:
        info = images_info[name]
        K = K_cache[name]
        R, t = info["R"], info["t"]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        proj = P @ h  # 3xN
        depth = proj[2]
        safe_d = np.where(depth > 0, depth, 1.0)
        u, v = proj[0] / safe_d, proj[1] / safe_d
        in_view = ((depth > 0) & (u >= 0) & (u < actual_w)
                   & (v >= 0) & (v < actual_h))
        visible += in_view.astype(int)
    return visible >= min_views


def multiview_color_blend(pts3d, valid_images, K_cache, images_info,
                          img_dir, actual_w, actual_h):
    """从所有可见视角混合提取颜色（比单视角更稳健）。"""
    n = len(pts3d)
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    color_sum = np.zeros((n, 3), dtype=np.float64)
    color_cnt = np.zeros(n, dtype=int)
    h = np.hstack([pts3d, np.ones((n, 1))]).T
    for name in valid_images:
        info = images_info[name]
        K = K_cache[name]
        R, t = info["R"], info["t"]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        proj = P @ h
        depth = proj[2]
        safe_d = np.where(depth > 0, depth, 1.0)
        u = np.round(proj[0] / safe_d).astype(int)
        v = np.round(proj[1] / safe_d).astype(int)
        valid = ((depth > 0) & (u >= 0) & (u < actual_w)
                 & (v >= 0) & (v < actual_h))
        if not np.any(valid):
            continue
        img = cv2.imread(str(img_dir / name))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        idx = np.where(valid)[0]
        color_sum[idx] += img_rgb[v[idx], u[idx]].astype(np.float64)
        color_cnt[idx] += 1
    color_cnt = np.maximum(color_cnt, 1)
    return (color_sum / color_cnt[:, None]).astype(np.uint8)


# ======================== 点云去噪 ========================

def statistical_outlier_removal(points, colors, k=10, std_mul=3.0):
    """统计离群值去除，保护稀疏簇（花瓣）。

    策略：先识别稀疏点（近邻距离 > 75th percentile 的 3 倍），
    在密集区域做 SOR，稀疏点全部保留。
    这样浮在空中的花瓣不会被删除。
    """
    if len(points) < k + 1:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)

    # 识别稀疏点：近邻距离远超过典型值的点是漂浮物（花瓣）
    p75 = np.percentile(mean_d, 75)
    is_sparse = mean_d > p75 * 3.0  # 孤立的稀疏点

    # 只对密集区域做 SOR
    dense_mask = ~is_sparse
    if dense_mask.sum() > k + 1:
        dense_mean = mean_d[dense_mask]
        thr = dense_mean.mean() + std_mul * dense_mean.std()
        # 密集区域内超过阈值的点删除
        dense_outlier = dense_mask & (mean_d > thr)
        keep = ~dense_outlier
    else:
        keep = np.ones(len(points), dtype=bool)

    # 稀疏点全部保留
    keep[is_sparse] = True
    return points[keep], colors[keep]


# ======================== 密度感知降采样 ========================

def density_aware_downsample(points, colors, target_n, density_k=16):
    """密度感知降采样：密集区域多删点，稀疏区域（如花瓣）几乎不删。

    策略：
    1. 用 KDTree 计算每个点的局部密度（k 近邻平均距离的倒数）
    2. 采样概率 ∝ 1/density（稀疏点概率高，密集点概率低）
    3. 按概率采样 target_n 个点
    """
    if len(points) <= target_n:
        return points, colors

    n = len(points)
    # 计算局部密度：k 近邻平均距离越小 → 密度越高
    tree = cKDTree(points)
    k = min(density_k, n - 1)
    dists, _ = tree.query(points, k=k + 1)  # 第 0 列是自身(距离=0)
    mean_dist = dists[:, 1:].mean(axis=1)    # 每个点到 k 近邻的平均距离

    # 避免除零
    mean_dist = np.maximum(mean_dist, 1e-10)

    # 采样权重 = 平均距离（稀疏的点距离大 → 权重大 → 更容易被保留）
    weights = mean_dist.copy()

    # 对最稀疏的 10% 点额外加权，确保花瓣等漂浮物被保留
    sparse_threshold = np.percentile(weights, 90)
    weights[weights >= sparse_threshold] *= 3.0

    # 归一化为概率
    prob = weights / weights.sum()

    rng = np.random.default_rng(42)
    chosen = rng.choice(n, size=target_n, replace=False, p=prob)
    return points[chosen], colors[chosen]


# ======================== 颜色提取（单视角，用于中间阶段） ========================

def extract_colors(img_path, K, R, t, pts3d):
    """将三维点投影到图像上获取 RGB 颜色。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((len(pts3d), 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
    proj = P @ h
    uv = (proj[:2] / proj[2:]).T
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    colors = np.zeros((len(pts3d), 3), dtype=np.uint8)
    colors[valid] = img_rgb[v[valid], u[valid]]
    return colors


# ======================== PLY 保存 ========================

def save_ply(filepath, points, colors=None):
    """保存点云为 .ply 文件（带 RGB 颜色）。"""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
    )
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(np.array(points[i], dtype=np.float32).tobytes())
            if has_color:
                f.write(np.array(colors[i], dtype=np.uint8).tobytes())


# ======================== 配对策略 ========================

def build_pairs(n_images, neighbor_range):
    """生成相邻相机配对列表（不循环）。"""
    pairs = []
    for i in range(n_images):
        for j in range(i + 1, min(i + 1 + neighbor_range, n_images)):
            pairs.append((i, j))
    return pairs


# ======================== 单帧处理 ========================

def process_frame(frame_id, frame_dir, cameras, images_info, model, cfg):
    """处理单帧：四阶段高可靠性 pipeline。

    Phase 1: 两两匹配 → RANSAC → 三角化 → 角度/深度/重投影严格过滤
    Phase 2: 体素合并去重 → 观测次数过滤（多对独立三角化确认）
    Phase 3: 多视角可见性验证
    Phase 4: 多视角颜色混合
    """
    img_dir = frame_dir / "images"

    available = sorted(
        [f.name for f in img_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=lambda x: int(Path(x).stem),
    )
    valid_images = [n for n in available if n in images_info]
    if len(valid_images) < 2:
        print(f"  [帧 {frame_id}] 有效图像不足 (< 2)，跳过")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    sample_img = PILImage.open(str(img_dir / valid_images[0]))
    actual_w, actual_h = sample_img.size

    K_cache = {}
    for name in valid_images:
        info = images_info[name]
        cam = cameras[info["camera_id"]]
        K_cache[name] = scale_intrinsics(cam["K"], cam["width"], cam["height"],
                                         actual_w, actual_h)

    pairs = build_pairs(len(valid_images), cfg.neighbor_range)

    # ==================== Phase 1: 逐对严格三角化 ====================
    raw_pts, raw_clr = [], []
    total_raw = 0
    pair_bar = tqdm(pairs, desc=f"  帧 {frame_id} Phase1", leave=False)

    for idx_i, idx_j in pair_bar:
        ni, nj = valid_images[idx_i], valid_images[idx_j]
        path_i, path_j = str(img_dir / ni), str(img_dir / nj)
        Ki, Kj = K_cache[ni], K_cache[nj]
        Ri, ti = images_info[ni]["R"], images_info[ni]["t"]
        Rj, tj = images_info[nj]["R"], images_info[nj]["t"]

        try:
            preds = model.match(path_i, path_j)
        except Exception as e:
            tqdm.write(f"    匹配失败 ({ni}<->{nj}): {e}")
            continue

        matches, overlaps, _, _ = model.sample(preds, cfg.num_samples)
        kpA, kpB = model.to_pixel_coordinates(matches, actual_h, actual_w,
                                               actual_h, actual_w)
        pts_i = kpA.cpu().numpy()
        pts_j = kpB.cpu().numpy()
        conf = overlaps.cpu().numpy()

        cmask = conf > cfg.min_confidence
        pts_i, pts_j = pts_i[cmask], pts_j[cmask]
        n_match = len(pts_i)
        if n_match < 8:
            continue

        # 用已知位姿计算 Fundamental Matrix，用极线距离过滤
        # 不用 RANSAC —— 避免背景主导导致花瓣匹配被报废
        F = compute_fundamental_from_poses(Ki, Ri, ti, Kj, Rj, tj)
        epi_mask = epipolar_filter(pts_i, pts_j, F, cfg.epipolar_threshold)
        in_i, in_j = pts_i[epi_mask], pts_j[epi_mask]
        n_inlier = len(in_i)
        if n_inlier < 10:
            continue

        pts3d = triangulate_points(Ki, Ri, ti, Kj, Rj, tj, in_i, in_j)

        # 三角化角度过滤：角度太小时深度极不稳定
        angles = compute_triangulation_angles(Ri, ti, Rj, tj, pts3d)
        amask = angles >= cfg.min_angle
        pts3d, in_i, in_j = pts3d[amask], in_i[amask], in_j[amask]

        # 深度过滤
        dmask = filter_by_depth(Ri, ti, Rj, tj, pts3d, cfg.min_depth, cfg.max_depth)
        pts3d, in_i, in_j = pts3d[dmask], in_i[dmask], in_j[dmask]
        if len(pts3d) == 0:
            continue

        # 严格重投影误差过滤
        rmask = filter_by_reprojection(Ki, Ri, ti, Kj, Rj, tj,
                                       in_i, in_j, pts3d, cfg.reproj_threshold)
        pts3d, in_i = pts3d[rmask], in_i[rmask]
        if len(pts3d) == 0:
            continue

        clr = extract_colors(path_i, Ki, Ri, ti, pts3d)
        raw_pts.append(pts3d)
        raw_clr.append(clr)
        total_raw += len(pts3d)

        tqdm.write(f"    [{ni}<->{nj}] 匹配:{n_match} 内点:{n_inlier} "
                   f"角度ok:{int(amask.sum())} 三角化:{len(pts3d)}")

    if not raw_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    all_pts = np.vstack(raw_pts)
    all_clr = np.vstack(raw_clr)
    tqdm.write(f"  [帧 {frame_id}] Phase1: {total_raw} 个原始点")

    # ==================== Phase 2: 体素合并 + 观测次数过滤 ====================
    merged_pts, merged_clr, obs = merge_and_count_observations(all_pts, all_clr)
    obs_mask = obs >= cfg.min_observations
    merged_pts, merged_clr = merged_pts[obs_mask], merged_clr[obs_mask]
    tqdm.write(f"  [帧 {frame_id}] Phase2: 合并 {total_raw} -> {len(merged_pts)} "
               f"(obs >= {cfg.min_observations})")

    if len(merged_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    # ==================== Phase 3: 多视角可见性验证 ====================
    vis_mask = multiview_visibility_filter(
        merged_pts, valid_images, K_cache, images_info,
        actual_w, actual_h, cfg.min_visible_views)
    merged_pts, merged_clr = merged_pts[vis_mask], merged_clr[vis_mask]
    tqdm.write(f"  [帧 {frame_id}] Phase3: 可见性 -> {len(merged_pts)} "
               f"(>= {cfg.min_visible_views} views)")

    if len(merged_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    # ==================== Phase 4: 多视角颜色混合 ====================
    tqdm.write(f"  [帧 {frame_id}] Phase4: 多视角颜色混合...")
    colors = multiview_color_blend(
        merged_pts, valid_images, K_cache, images_info,
        img_dir, actual_w, actual_h)

    return merged_pts, colors


# ======================== 主函数 ========================

def main():
    parser = argparse.ArgumentParser(
        description="多相机三维点云重建 (RoMa v2)")
    parser.add_argument("--data_dir", default="images_processed",
                        help="数据根目录")
    parser.add_argument("--output_dir", default="output",
                        help="输出目录")
    parser.add_argument("--frames", default=None,
                        help="逗号分隔的帧号，如 1,2,3 (默认全部)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="每对图像采样匹配数")
    parser.add_argument("--target_points", type=int, default=50000,
                        help="每帧最终目标点数")
    parser.add_argument("--neighbor_range", type=int, default=3,
                        help="相邻相机配对范围")
    parser.add_argument("--roma_setting", default="fast",
                        choices=["turbo", "fast", "base", "precise"],
                        help="RoMa v2 速度/精度模式")
    parser.add_argument("--min_confidence", type=float, default=0.1,
                        help="最小 overlap 置信度")
    parser.add_argument("--ransac_threshold", type=float, default=1.0,
                        help="RANSAC 阈值 (像素，备用)")
    parser.add_argument("--reproj_threshold", type=float, default=2.0,
                        help="逐对重投影误差阈值 (像素)")
    parser.add_argument("--epipolar_threshold", type=float, default=2.0,
                        help="极线距离阈值 (像素)")
    parser.add_argument("--max_depth", type=float, default=100.0,
                        help="最大深度")
    parser.add_argument("--min_angle", type=float, default=1.0,
                        help="最小三角化角度 (度)")
    parser.add_argument("--min_observations", type=int, default=1,
                        help="点至少被多少对相机独立三角化")
    parser.add_argument("--min_visible_views", type=int, default=2,
                        help="点至少被多少个相机看到")
    parser.add_argument("--sor_k", type=int, default=10,
                        help="SOR 近邻数")
    parser.add_argument("--sor_std", type=float, default=3.0,
                        help="SOR 标准差倍数")
    parser.add_argument("--compile", action="store_true",
                        help="启用 torch.compile (需要 Triton)")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_points=args.target_points,
        min_confidence=args.min_confidence,
        ransac_threshold=args.ransac_threshold,
        reproj_threshold=args.reproj_threshold,
        epipolar_threshold=args.epipolar_threshold,
        max_depth=args.max_depth,
        min_angle=args.min_angle,
        min_observations=args.min_observations,
        min_visible_views=args.min_visible_views,
        neighbor_range=args.neighbor_range,
        sor_k=args.sor_k,
        sor_std=args.sor_std,
        roma_setting=args.roma_setting,
        compile=args.compile,
    )

    data_path = Path(cfg.data_dir)
    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. 解析 COLMAP 相机参数 ----
    sparse_dir = data_path / "sparse" / "0"
    print(f"[1/4] 解析相机参数: {sparse_dir}")
    cameras = parse_cameras(sparse_dir / "cameras.txt")
    images_info = parse_images(sparse_dir / "images.txt")
    print(f"      相机数: {len(cameras)}, COLMAP 图像数: {len(images_info)}")

    # ---- 2. 加载 RoMa v2 ----
    print(f"[2/4] 加载 RoMa v2  (setting={cfg.roma_setting}, compile={cfg.compile})")
    model = RoMaV2(RoMaV2.Cfg(compile=cfg.compile))
    model.apply_setting(cfg.roma_setting)
    dev = next(model.parameters()).device
    print(f"      模型就绪，设备: {dev}")

    # ---- 3. 确定帧列表 ----
    if args.frames:
        frame_ids = [s.strip() for s in args.frames.split(",")]
    else:
        frame_ids = sorted(
            [d.name for d in data_path.iterdir()
             if d.is_dir() and d.name != "sparse"],
            key=lambda x: int(x),
        )
    print(f"[3/4] 待处理帧: {len(frame_ids)} 帧")
    print(f"      参数: samples={cfg.num_samples}, target={cfg.target_points}, "
          f"neighbors={cfg.neighbor_range}, "
          f"ransac_thr={cfg.ransac_threshold}, reproj_thr={cfg.reproj_threshold}")

    # ---- 4. 逐帧处理 ----
    print("[4/4] 开始重建 ...")
    for fid in tqdm(frame_ids, desc="总进度"):
        fdir = data_path / fid
        if not (fdir / "images").exists():
            tqdm.write(f"  [帧 {fid}] 目录不存在，跳过")
            continue

        points, colors = process_frame(fid, fdir, cameras, images_info, model, cfg)

        if len(points) == 0:
            tqdm.write(f"  [帧 {fid}] 无有效三维点")
            continue

        before = len(points)

        # 统计离群值去除
        points, colors = statistical_outlier_removal(
            points, colors, k=cfg.sor_k, std_mul=cfg.sor_std)
        after_sor = len(points)

        # 密度感知降采样：密集区域多删，稀疏区域（花瓣）保留
        points, colors = density_aware_downsample(points, colors, cfg.target_points)

        ply_path = out_path / f"pcd{fid}.ply"
        save_ply(ply_path, points, colors)
        tqdm.write(f"  [帧 {fid}] 点数: {before} -> {after_sor}(去噪) -> {len(points)}(降采样)  "
                   f"=> {ply_path}")

    print("\n全部完成！输出目录:", out_path.resolve())


if __name__ == "__main__":
    main()
