"""Run COLMAP reconstruction with SuperGlue matches instead of COLMAP matcher.

The script accepts a multi-camera frame layout:

  data/twopeople/images/<frame_dir>/<camera_image>.png

or a flat single-frame folder containing the camera images directly.  A
camera-grid JSON exported by ``camera_grid_layout_tool.py`` can restrict
matching to manually confirmed physical neighbours.

It creates a COLMAP database, writes SuperPoint+SuperGlue keypoints/matches to
the database, then runs COLMAP mapper/model_converter.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
from collections import Counter, deque
import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


MAX_IMAGE_ID = 2147483647
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
}

LOGGER = logging.getLogger("superglue_colmap")


@dataclass(frozen=True)
class ImageInfo:
    name: str
    path: Path
    width: int
    height: int
    image_id: int
    camera_id: int


@dataclass(frozen=True)
class InputImage:
    """One source image and the stable camera filename used by COLMAP."""

    path: Path
    output_name: str


@dataclass(frozen=True)
class CameraMajorFrame:
    """Virtual frame assembled from camera-major input folders."""

    name: str
    images: tuple[InputImage, ...]


@dataclass(frozen=True)
class ManualCameraLayout:
    """Camera graph exported by the interactive physical-layout tool."""

    source_path: Path
    camera_ids: tuple[str, ...]
    placed_ids: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    neighbour_mode: int | None = None
    neighbour_radius: int | None = None
    positions: tuple[tuple[int, int, tuple[str, ...]], ...] = ()


FrameInput = Path | CameraMajorFrame


@dataclass(frozen=True)
class PairMatch:
    name0: str
    name1: str
    matches: np.ndarray  # Nx4 original-image coordinates: x0, y0, x1, y1


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray
    rotmat: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: list[tuple[int, int]]


@dataclass(frozen=True)
class VelocityResult:
    velocities: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    view_counts: np.ndarray


class CommandError(RuntimeError):
    pass


def natural_key(path_or_name: Path | str) -> tuple:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    stem = Path(name).stem
    if stem.isdigit():
        return (0, int(stem), name)
    parts: list[tuple[int, int | str]] = []
    chunk = ""
    is_digit = stem[:1].isdigit()
    for ch in stem:
        if ch.isdigit() == is_digit:
            chunk += ch
        else:
            parts.append((0, int(chunk)) if is_digit else (1, chunk))
            chunk = ch
            is_digit = ch.isdigit()
    if chunk:
        parts.append((0, int(chunk)) if is_digit else (1, chunk))
    return (1, parts, name)


class _TqdmLoggingHandler(logging.Handler):
    """Emit log records via tqdm.write so live progress bars are not shredded."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record))
        except Exception:
            try:
                print(self.format(record))
            except Exception:
                self.handleError(record)


class _NullBar:
    """No-op stand-in for tqdm when progress is disabled or tqdm is missing."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, *a, **k) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


def make_pbar(total: int | None, desc: str, unit: str, enable: bool, leave: bool = True, position: int = 0):
    """A tqdm bar when enabled+available, else a silent no-op with the same interface."""
    if enable:
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave, position=position)
        except Exception:
            pass
    return _NullBar()


def setup_logging(verbose: bool, progress: bool = True) -> None:
    # Stream output line-by-line so logs/bars appear promptly even through a pipe.
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    handler: logging.Handler = _TqdmLoggingHandler() if progress else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)


def auto_workers(kind: str) -> int:
    """Pick a sensible default worker count from the CPU core count."""
    cores = os.cpu_count() or 4
    if kind == "solver":
        # Disk-bound COLMAP undistort: 2 concurrent solves usually saturate disk without thrash.
        return 2 if cores >= 8 else 1
    # crop export (cv2, GIL-released): more threads overlap disk, capped to avoid contention.
    return max(1, min(cores, 8))


def parse_gpus(spec: str | None) -> list[int]:
    """Parse ``--gpus`` into a list of physical device indices.

    '' / None -> [] (single-GPU / current behavior). 'all' -> every visible CUDA
    device. '0,1,2,3' -> [0, 1, 2, 3]. Duplicates are dropped, order preserved.
    """
    if not spec:
        return []
    spec = spec.strip()
    if spec.lower() == "all":
        try:
            import torch

            n = int(torch.cuda.device_count())
        except Exception:
            n = 0
        return list(range(n))
    out: list[int] = []
    seen: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        idx = int(tok)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def resolve_colmap(colmap_arg: str) -> str:
    # Prefer a repository-local Windows bundle for the default value.  Conda often
    # installs a tiny ``Scripts/colmap.BAT`` shim whose hard-coded target becomes
    # stale when the repository is moved or the bundle is only partly extracted.
    if colmap_arg == "colmap" and sys.platform == "win32":
        tools_dir = Path(__file__).resolve().parents[1] / "tools"
        bundled = sorted(
            tools_dir.glob("colmap-*/COLMAP.bat"),
            key=lambda path: natural_key(path.parent.name),
            reverse=True,
        )
        for candidate in bundled:
            exe = candidate.parent / "bin" / "colmap.exe"
            if exe.is_file():
                return str(candidate.resolve())

    colmap = shutil.which(colmap_arg) if os.path.basename(colmap_arg) == colmap_arg else colmap_arg
    if not colmap:
        raise FileNotFoundError(
            "COLMAP was not found. Install it under tools/colmap-*/ or use "
            "--colmap to point to COLMAP.bat/colmap.exe."
        )

    path = Path(colmap).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP launcher does not exist: {path}")
    return str(path)


def run_command(cmd: Sequence[str], log_path: Path | None = None) -> None:
    LOGGER.info("run: %s", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    if sys.platform == "win32":
        exe_dir = str(Path(cmd[0]).resolve().parent)
        conda_prefix = env.get("CONDA_PREFIX", "")
        prepend = [exe_dir]
        if conda_prefix:
            prepend.extend(
                str(Path(conda_prefix) / p)
                for p in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "Scripts")
            )
        env["PATH"] = os.pathsep.join(prepend) + os.pathsep + env.get("PATH", "")

    started = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed = time.time() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            " ".join(str(x) for x in cmd)
            + "\n\n[stdout]\n"
            + result.stdout
            + "\n\n[stderr]\n"
            + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-30:])
        raise CommandError(f"command failed with exit={result.returncode}: {' '.join(cmd)}\n{tail}")
    LOGGER.info("done in %.1fs", elapsed)


def discover_frame_dirs(images_root: Path) -> list[Path]:
    if not images_root.exists():
        raise FileNotFoundError(f"images root does not exist: {images_root}")
    frames = sorted([p for p in images_root.iterdir() if p.is_dir()], key=natural_key)
    if not frames:
        direct_images = [
            path for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(direct_images) >= 2:
            LOGGER.info(
                "flat single-frame input: using %d images directly under %s",
                len(direct_images), images_root,
            )
            return [images_root]
        raise FileNotFoundError(
            f"no frame folders or flat image set (at least two images) found in {images_root}"
        )
    return frames


def discover_camera_major_frames(images_root: Path) -> tuple[list[CameraMajorFrame], dict[str, str]]:
    """Transpose ``camera/frame`` input into virtual ``frame/camera.png`` frames.

    Camera folders and images inside each folder are naturally sorted. All
    cameras must contain the same frame stems in the same order; count-only
    alignment can silently mix capture times and corrupt calibration.
    """
    if not images_root.exists():
        raise FileNotFoundError(f"images root does not exist: {images_root}")
    camera_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()], key=natural_key)
    if len(camera_dirs) < 2:
        raise ValueError(f"camera-major input needs at least two camera folders in {images_root}")

    images_by_camera: list[list[Path]] = []
    frame_stems_by_camera: list[list[str]] = []
    for camera_dir in camera_dirs:
        images = sorted(
            [p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
            key=natural_key,
        )
        if not images:
            raise ValueError(f"camera folder contains no images: {camera_dir}")
        frame_stems = [p.stem for p in images]
        stem_counts = Counter(frame_stems)
        if len(stem_counts) != len(frame_stems):
            duplicates = sorted((stem for stem, count in stem_counts.items() if count > 1), key=natural_key)
            raise ValueError(
                f"camera folder has duplicate frame stems (usually mixed extensions): "
                f"{camera_dir}: {', '.join(duplicates[:10])}"
            )
        images_by_camera.append(images)
        frame_stems_by_camera.append(frame_stems)

    counts = [len(images) for images in images_by_camera]
    if len(set(counts)) != 1:
        details = ", ".join(f"{d.name}={n}" for d, n in zip(camera_dirs, counts))
        raise ValueError(f"camera folders have different frame counts: {details}")

    # Equal counts are not enough: a missing frame plus an unrelated extra frame
    # would otherwise shift every later capture and silently mix timestamps.
    reference_stems = frame_stems_by_camera[0]
    reference_set = set(reference_stems)
    for camera_dir, stems in zip(camera_dirs[1:], frame_stems_by_camera[1:]):
        if stems != reference_stems:
            stem_set = set(stems)
            missing = sorted(reference_set - stem_set, key=natural_key)
            extra = sorted(stem_set - reference_set, key=natural_key)
            detail_parts = []
            if missing:
                detail_parts.append(f"missing={','.join(missing[:10])}")
            if extra:
                detail_parts.append(f"extra={','.join(extra[:10])}")
            if not detail_parts:
                detail_parts.append("frame order differs")
            raise ValueError(
                "camera folders must contain the same frame stems in the same natural order; "
                f"{camera_dir.name} differs from {camera_dirs[0].name} "
                f"({'; '.join(detail_parts)})"
            )

    mapping = {f"{index}.png": camera_dir.name for index, camera_dir in enumerate(camera_dirs, 1)}
    frames = [
        CameraMajorFrame(
            name=str(frame_index + 1),
            images=tuple(
                InputImage(images[frame_index], f"{camera_index}.png")
                for camera_index, images in enumerate(images_by_camera, 1)
            ),
        )
        for frame_index in range(counts[0])
    ]
    return frames, mapping


def parse_frame_selection(frame_dirs: Sequence[FrameInput], frames_arg: str | None) -> list[FrameInput]:
    if not frames_arg:
        return list(frame_dirs)
    by_name = {p.name: p for p in frame_dirs}
    selected: list[FrameInput] = []
    for token in [x.strip() for x in frames_arg.split(",") if x.strip()]:
        if ":" in token:
            start_s, end_s = token.split(":", 1)
            for value in range(int(start_s), int(end_s) + 1):
                key = str(value)
                if key in by_name:
                    selected.append(by_name[key])
        elif token in by_name:
            selected.append(by_name[token])
        elif token.isdigit() and 1 <= int(token) <= len(frame_dirs):
            selected.append(frame_dirs[int(token) - 1])
        else:
            raise ValueError(f"frame '{token}' not found")
    unique: list[FrameInput] = []
    seen: set[FrameInput] = set()
    for frame in selected:
        if frame not in seen:
            unique.append(frame)
            seen.add(frame)
    if not unique:
        raise ValueError("--frames selected no existing frame folders")
    return unique


def list_images(frame_dir: FrameInput, max_images: int | None = None) -> list[InputImage]:
    if isinstance(frame_dir, CameraMajorFrame):
        images = list(frame_dir.images)
    else:
        images = [
            InputImage(p, p.name)
            for p in sorted(
                [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
                key=natural_key,
            )
        ]
    if max_images is not None:
        images = images[:max_images]
    if len(images) < 2:
        raise ValueError(f"need at least two images in {frame_dir}, got {len(images)}")
    return images


def image_size(path: Path) -> tuple[int, int]:
    # PIL reads only the header (no 4K decode); fall back to cv2 if PIL is absent.
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read image size: {path}")
    return int(img.shape[1]), int(img.shape[0])


def stage_images(src_images: Sequence[InputImage], image_dir: Path, copy_images: bool) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for item in src_images:
        src = item.path
        dst = image_dir / item.output_name
        if dst.exists():
            dst.unlink()
        # Camera-major inputs are normalized to N.png.  Re-encode only when the
        # source is not already PNG; otherwise preserve bytes via copy/hard link.
        if src.suffix.lower() != dst.suffix.lower():
            import cv2

            image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if image is None or not cv2.imwrite(str(dst), image):
                raise RuntimeError(f"failed to convert image to PNG: {src}")
        elif copy_images:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def camera_params(model: str, width: int, height: int, focal_factor: float) -> np.ndarray:
    focal = focal_factor * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    if model == "SIMPLE_PINHOLE":
        return np.asarray([focal, cx, cy], dtype=np.float64)
    if model == "PINHOLE":
        return np.asarray([focal, focal, cx, cy], dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.asarray([focal, cx, cy, 0.0], dtype=np.float64)
    if model == "RADIAL":
        return np.asarray([focal, cx, cy, 0.0, 0.0], dtype=np.float64)
    if model == "OPENCV":
        return np.asarray([focal, focal, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"unsupported camera model: {model}")


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def array_to_blob(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def reset_database(colmap: str, database_path: Path, log_dir: Path) -> None:
    if database_path.exists():
        database_path.unlink()
    run_command([colmap, "database_creator", "--database_path", str(database_path)], log_dir / "database_creator.log")


def create_colmap_database(
    database_path: Path,
    images: Sequence[Path],
    camera_model: str,
    focal_factor: float,
    single_camera: bool,
) -> dict[str, ImageInfo]:
    con = sqlite3.connect(database_path)
    try:
        model_id = CAMERA_MODEL_IDS[camera_model]
        image_infos: dict[str, ImageInfo] = {}
        shared_camera_id: int | None = None
        shared_size: tuple[int, int] | None = None
        for image_path in images:
            width, height = image_size(image_path)
            if single_camera:
                if shared_camera_id is None:
                    shared_size = (width, height)
                    params = camera_params(camera_model, width, height, focal_factor)
                    cur = con.execute(
                        "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                        (model_id, width, height, array_to_blob(params), 0),
                    )
                    shared_camera_id = int(cur.lastrowid)
                elif shared_size != (width, height):
                    raise ValueError("--single_camera requires all images to have identical sizes")
                camera_id = shared_camera_id
            else:
                params = camera_params(camera_model, width, height, focal_factor)
                cur = con.execute(
                    "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                    (model_id, width, height, array_to_blob(params), 0),
                )
                camera_id = int(cur.lastrowid)

            cur = con.execute("INSERT INTO images(name, camera_id) VALUES (?, ?)", (image_path.name, camera_id))
            image_id = int(cur.lastrowid)
            image_infos[image_path.name] = ImageInfo(image_path.name, image_path, width, height, image_id, camera_id)
        con.commit()
        return image_infos
    finally:
        con.close()


def build_pairs(
    image_names: Sequence[str],
    mode: str,
    window: int,
    loop: bool,
    pairs_file: Path | None = None,
) -> list[tuple[str, str]]:
    if mode == "exhaustive":
        pairs = list(itertools.combinations(image_names, 2))
    elif mode == "sequential":
        pairs = []
        n = len(image_names)
        for i in range(n):
            for step in range(1, min(window, n - 1) + 1):
                j = i + step
                if j < n:
                    pairs.append((image_names[i], image_names[j]))
                elif loop:
                    pairs.append((image_names[i], image_names[j % n]))
    elif mode == "pairs_file":
        if not pairs_file:
            raise ValueError("--pairs_file is required when --pair_mode pairs_file")
        name_set = set(image_names)
        pairs = []
        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2 or parts[0] not in name_set or parts[1] not in name_set:
                raise ValueError(f"invalid pair line: {line}")
            pairs.append((parts[0], parts[1]))
    else:
        raise ValueError(f"unsupported pair mode: {mode}")

    return normalize_pairs(pairs, image_names)


def normalize_pairs(pairs: Sequence[tuple[str, str]], image_names: Sequence[str]) -> list[tuple[str, str]]:
    """Dedupe and orient pairs by the image order (a before b)."""
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    order = {name: idx for idx, name in enumerate(image_names)}
    for a, b in pairs:
        if a == b:
            continue
        pair = (a, b) if order[a] < order[b] else (b, a)
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return normalized


def _slot_camera_ids(slot: object, context: str) -> tuple[bool, list[str]]:
    """Read both version-1 ``camera_id`` and version-2 ``camera_ids`` slots."""
    if slot is None:
        return False, []
    if not isinstance(slot, dict):
        raise ValueError(f"{context}: every grid position must be an object or null")
    enabled = slot.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{context}: grid position 'enabled' must be true or false")
    if "camera_ids" in slot:
        values = slot["camera_ids"]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{context}: camera_ids must be a list of non-empty strings")
        camera_ids = list(values)
    else:
        legacy = slot.get("camera_id")
        camera_ids = [str(legacy)] if legacy else []
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError(f"{context}: a physical position contains a camera more than once")
    if not enabled and camera_ids:
        raise ValueError(f"{context}: a disabled physical gap cannot contain cameras")
    return enabled, camera_ids


def _pairs_from_grid_positions(
    positions: Sequence[tuple[int, int, Sequence[str]]],
    mode: int,
    radius: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for _row, _column, camera_ids in positions:
        pairs.extend(itertools.combinations(camera_ids, 2))
    for (row0, col0, cameras0), (row1, col1, cameras1) in itertools.combinations(positions, 2):
        dr, dc = abs(row0 - row1), abs(col0 - col1)
        adjacent = (
            ((dr == 0 and 0 < dc <= radius) or (dc == 0 and 0 < dr <= radius))
            if mode == 4
            else max(dr, dc) <= radius and bool(dr or dc)
        )
        if adjacent:
            pairs.extend((camera0, camera1) for camera0 in cameras0 for camera1 in cameras1)

    unique: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for camera0, camera1 in pairs:
        if camera0 == camera1:
            continue
        key = frozenset((camera0, camera1))
        if key not in seen:
            seen.add(key)
            unique.append((camera0, camera1))
    return unique


def load_manual_camera_layout(path: Path) -> ManualCameraLayout:
    """Load camera-grid JSON or its exported neighbour-pairs text file."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"camera layout file does not exist: {path}")
    if path.suffix.lower() != ".json":
        camera_ids: list[str] = []
        seen_ids: set[str] = set()
        pairs: list[tuple[str, str]] = []
        seen_pairs: set[frozenset[str]] = set()
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected two whitespace-separated camera IDs"
                )
            camera0, camera1 = parts
            if camera0 == camera1:
                raise ValueError(f"{path}:{line_number}: a camera cannot be paired with itself")
            key = frozenset((camera0, camera1))
            if key not in seen_pairs:
                pairs.append((camera0, camera1))
                seen_pairs.add(key)
            for camera_id in parts:
                if camera_id not in seen_ids:
                    camera_ids.append(camera_id)
                    seen_ids.add(camera_id)
        if not pairs:
            raise ValueError(f"camera layout pair file contains no usable pairs: {path}")
        ids = tuple(camera_ids)
        return ManualCameraLayout(path, ids, ids, tuple(pairs))

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid camera layout JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != "camera-grid-layout":
        raise ValueError(f"not a camera-grid-layout JSON file: {path}")
    version = data.get("version")
    if version not in {1, 2}:
        raise ValueError(f"unsupported camera layout version {version!r}: {path}")

    raw_cameras = data.get("cameras", [])
    if not isinstance(raw_cameras, list):
        raise ValueError(f"camera layout 'cameras' must be a list: {path}")
    camera_ids: list[str] = []
    seen_cameras: set[str] = set()
    for item in raw_cameras:
        if not isinstance(item, dict) or not isinstance(item.get("camera_id"), str) or not item["camera_id"]:
            raise ValueError(f"camera layout contains an invalid camera item: {path}")
        camera_id = item["camera_id"]
        if camera_id in seen_cameras:
            raise ValueError(f"camera layout contains duplicate camera ID {camera_id!r}: {path}")
        camera_ids.append(camera_id)
        seen_cameras.add(camera_id)

    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"camera layout contains no grid rows: {path}")
    positions: list[tuple[int, int, Sequence[str]]] = []
    placed_ids: list[str] = []
    seen_placed: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or not row:
            raise ValueError(f"camera layout row {row_index + 1} is empty or invalid: {path}")
        for column_index, slot in enumerate(row):
            enabled, slot_ids = _slot_camera_ids(
                slot, f"{path} row {row_index + 1} column {column_index + 1}"
            )
            for camera_id in slot_ids:
                if camera_id not in seen_cameras:
                    raise ValueError(f"grid position refers to unknown camera {camera_id!r}: {path}")
                if camera_id in seen_placed:
                    raise ValueError(f"camera appears in more than one grid position: {camera_id!r}")
                seen_placed.add(camera_id)
                placed_ids.append(camera_id)
            if enabled and slot_ids:
                positions.append((row_index, column_index, tuple(slot_ids)))

    settings = data.get("neighbour_settings", {})
    if not isinstance(settings, dict):
        raise ValueError(f"camera layout neighbour_settings must be an object: {path}")
    mode = settings.get("mode", 8)
    radius = settings.get("radius", 1)
    if mode not in {4, 8}:
        raise ValueError(f"camera layout neighbour mode must be 4 or 8: {path}")
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError(f"camera layout neighbour radius must be a positive integer: {path}")
    pairs = _pairs_from_grid_positions(positions, mode, radius)
    if not pairs:
        raise ValueError(f"camera layout produces no neighbour pairs: {path}")
    return ManualCameraLayout(
        path,
        tuple(camera_ids),
        tuple(placed_ids),
        tuple(pairs),
        mode,
        radius,
        tuple((row, column, tuple(ids)) for row, column, ids in positions),
    )


def _camera_name_alias_levels(value: str) -> tuple[set[str], set[str], set[str]]:
    name = Path(value).name.casefold()
    stem = Path(name).stem
    stripped: set[str] = set()
    match = re.match(r"^\d{2,}[-_](.+)$", stem)
    if match and match.group(1):
        stripped.add(match.group(1))
    return {name}, {stem}, stripped


def resolve_manual_camera_names(
    image_names: Sequence[str],
    layout: ManualCameraLayout,
    camera_name_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve layout camera IDs to current frame image names.

    Resolution prefers exact file name, then stem, then a stable suffix obtained by
    removing a leading numeric frame token (``000001-03-color`` -> ``03-color``).
    Camera-major input additionally supplies the generated-name -> camera-folder map.
    """
    mapping = camera_name_mapping or {}
    actual_aliases: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for image_name in image_names:
        levels = [set(), set(), set()]
        labels = [image_name]
        if image_name in mapping:
            labels.append(mapping[image_name])
        for label in labels:
            for level, aliases in enumerate(_camera_name_alias_levels(label)):
                levels[level].update(aliases)
        actual_aliases[image_name] = (levels[0], levels[1], levels[2])

    resolved: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for camera_id in layout.camera_ids:
        manual_levels = _camera_name_alias_levels(camera_id)
        match_name: str | None = None
        for level, aliases in enumerate(manual_levels):
            if not aliases:
                continue
            candidates = [
                image_name for image_name, actual_levels in actual_aliases.items()
                if aliases & actual_levels[level]
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"camera layout ID {camera_id!r} ambiguously matches current images: "
                    + ", ".join(candidates)
                )
            if candidates:
                match_name = candidates[0]
                break
        if match_name is None:
            continue
        previous = claimed.get(match_name)
        if previous is not None and previous != camera_id:
            raise ValueError(
                f"camera layout IDs {previous!r} and {camera_id!r} both map to {match_name!r}"
            )
        resolved[camera_id] = match_name
        claimed[match_name] = camera_id
    return resolved


def build_manual_camera_pairs(
    image_names: Sequence[str],
    layout: ManualCameraLayout,
    camera_name_mapping: dict[str, str] | None = None,
    unlisted_policy: str = "error",
) -> list[tuple[str, str]]:
    """Resolve layout camera IDs to current frame names and build the candidate graph."""
    if unlisted_policy not in {"error", "exhaustive", "ignore"}:
        raise ValueError(f"unsupported camera-layout unlisted policy: {unlisted_policy}")
    resolved = resolve_manual_camera_names(image_names, layout, camera_name_mapping)

    raw_pairs = [
        (resolved[camera0], resolved[camera1])
        for camera0, camera1 in layout.pairs
        if camera0 in resolved and camera1 in resolved
    ]
    pairs = normalize_pairs(raw_pairs, image_names)
    paired_names = {name for pair in pairs for name in pair}
    uncovered = [name for name in image_names if name not in paired_names]
    unresolved_layout = [camera_id for camera_id in layout.camera_ids if camera_id not in resolved]
    if unresolved_layout:
        LOGGER.warning(
            "camera layout: %d layout camera(s) absent from this frame: %s",
            len(unresolved_layout),
            ", ".join(unresolved_layout[:10]) + ("..." if len(unresolved_layout) > 10 else ""),
        )
    if uncovered and unlisted_policy == "error":
        raise ValueError(
            "camera layout leaves current images unmapped or without any neighbour: "
            + ", ".join(uncovered[:20])
            + ("..." if len(uncovered) > 20 else "")
            + ". Fix the layout, or explicitly use --camera_layout_unlisted exhaustive/ignore."
        )
    if uncovered and unlisted_policy == "exhaustive":
        expanded = list(pairs)
        for image_name in uncovered:
            expanded.extend((image_name, other) for other in image_names if other != image_name)
        pairs = normalize_pairs(expanded, image_names)
        LOGGER.warning(
            "camera layout: %d uncovered image(s) use exhaustive fallback", len(uncovered)
        )
    elif uncovered:
        LOGGER.warning(
            "camera layout: ignoring %d image(s) without a manual neighbour: %s",
            len(uncovered), ", ".join(uncovered[:10]) + ("..." if len(uncovered) > 10 else ""),
        )
    if not pairs:
        raise ValueError("camera layout produced no usable pairs for the current frame")
    mapped_count = len(set(resolved.values()))
    LOGGER.info(
        "camera layout: mapped %d/%d images, %d covered by pairs -> %d manual candidate pairs",
        mapped_count, len(image_names), len(image_names) - len(uncovered), len(pairs),
    )
    return pairs


def manual_colocated_images(
    image_names: Sequence[str],
    layout: ManualCameraLayout | None,
    camera_name_mapping: dict[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return other images assigned to the same physical grid position."""
    if layout is None or not layout.positions:
        return {}
    resolved = resolve_manual_camera_names(image_names, layout, camera_name_mapping)
    colocated: dict[str, tuple[str, ...]] = {}
    for _row, _column, camera_ids in layout.positions:
        names = tuple(resolved[camera_id] for camera_id in camera_ids if camera_id in resolved)
        for name in names:
            others = tuple(other for other in names if other != name)
            if others:
                colocated[name] = others
    return colocated


def build_registration_rescue_pairs(
    image_names: Sequence[str],
    base_pairs: Sequence[tuple[str, str]],
    missing_names: Sequence[str],
    registered_names: set[str],
    hops: int = 2,
    max_pairs_per_image: int = 16,
) -> list[tuple[str, str]]:
    """Expand failed cameras through the view graph to registered cameras.

    With a radius-one grid, two graph hops approximate one additional physical
    ring.  Expanding only failed cameras avoids the cost and ambiguity of making
    the full camera graph denser.
    """
    if hops < 2 or max_pairs_per_image <= 0:
        return []
    normalized = normalize_pairs(base_pairs, image_names)
    adjacency = {name: set() for name in image_names}
    existing = {frozenset(pair) for pair in normalized}
    for name0, name1 in normalized:
        adjacency[name0].add(name1)
        adjacency[name1].add(name0)

    rescue: list[tuple[str, str]] = []
    for missing in sorted(set(missing_names), key=natural_key):
        if missing not in adjacency:
            continue
        distances = {missing: 0}
        frontier = {missing}
        for distance in range(1, hops + 1):
            next_frontier: set[str] = set()
            for current in frontier:
                for neighbour in adjacency[current]:
                    if neighbour not in distances:
                        distances[neighbour] = distance
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        direct = adjacency[missing]
        ranked = []
        for candidate, distance in distances.items():
            pair_key = frozenset((missing, candidate))
            if (
                distance < 2
                or candidate not in registered_names
                or candidate == missing
                or pair_key in existing
            ):
                continue
            shared_neighbours = len(direct & adjacency[candidate])
            ranked.append((distance, -shared_neighbours, natural_key(candidate), candidate))
        ranked.sort()
        rescue.extend((missing, entry[-1]) for entry in ranked[:max_pairs_per_image])
    return normalize_pairs(rescue, image_names)


def load_layout_file(path: Path) -> list[str]:
    """Read the physical camera order: one camera name (or comma list) per line, '#' comments.

    Entries are matched to images by stem, so '12', '12.png' and 'cam12.jpg' style
    names all work as long as the stems agree."""
    stems: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            stem = Path(token).stem
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
    if len(stems) < 2:
        raise ValueError(f"--layout_file needs at least two camera names: {path}")
    return stems


def build_layout_pairs(
    image_names: Sequence[str],
    layout_stems: Sequence[str],
    window: int,
    ring: bool,
) -> list[tuple[str, str]]:
    """Pairs restricted to cameras within `window` steps of each other in the
    user-provided physical order. This both avoids matching look-alike cameras on
    opposite sides of the rig and cuts the O(N^2) exhaustive pair count."""
    stem_to_name = {Path(n).stem: n for n in image_names}
    ordered = [stem_to_name[s] for s in layout_stems if s in stem_to_name]
    unknown_layout = [s for s in layout_stems if s not in stem_to_name]
    if unknown_layout:
        LOGGER.warning(
            "layout: %d layout entr%s did not match any image (check for typos): %s",
            len(unknown_layout), "y" if len(unknown_layout) == 1 else "ies",
            ", ".join(unknown_layout[:10]) + ("..." if len(unknown_layout) > 10 else ""),
        )
    layout_set = {Path(n).stem for n in ordered}
    missing = [n for n in image_names if Path(n).stem not in layout_set]
    if missing:
        LOGGER.warning(
            "layout: %d image(s) not listed in --layout_file are matched against every camera "
            "(slower but safe): %s",
            len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
        )

    pairs: list[tuple[str, str]] = []
    n = len(ordered)
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            if ring and n > 2:
                d = min(d, n - d)
            if d <= window:
                pairs.append((ordered[i], ordered[j]))
    for name in missing:
        for other in image_names:
            if other != name:
                pairs.append((name, other))
    return normalize_pairs(pairs, image_names)


def load_pair_blacklist(path: Path) -> set[frozenset[str]]:
    """Read 'nameA nameB' lines (stems, '#' comments) of pairs that must never be matched."""
    entries: set[frozenset[str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"invalid blacklist line (need two names): {line}")
        entries.add(frozenset((Path(parts[0]).stem, Path(parts[1]).stem)))
    return entries


def filter_blacklisted_pairs(
    pairs: Sequence[tuple[str, str]],
    blacklist: set[frozenset[str]],
) -> list[tuple[str, str]]:
    kept = [p for p in pairs if frozenset((Path(p[0]).stem, Path(p[1]).stem)) not in blacklist]
    if len(kept) != len(pairs):
        LOGGER.info("pair blacklist removed %d/%d pairs", len(pairs) - len(kept), len(pairs))
    return kept


def build_rig_pairs(
    ref_images: dict[int, "ColmapImage"],
    ref_points: Sequence["ColmapPoint3D"],
    names: Sequence[str],
    top_k: int,
    min_shared: int,
    geo_neighbors: int,
) -> list[tuple[str, str]]:
    """Pick the camera pairs worth matching for a static rig, computed once from the
    reference solve.

    Combines two signals so the set stays valid even as subjects move between frames:
      * covisibility -- cameras that share many reference 3D points;
      * geometry -- each camera's nearest neighbours by viewing angle around the scene
        centroid (covers overlaps that were textureless in the reference frame).
    """
    cam_index = {n: i for i, n in enumerate(names)}
    m = len(names)
    if m < 2:
        return []
    id_to_idx = {iid: cam_index[img.name] for iid, img in ref_images.items() if img.name in cam_index}

    covis = np.zeros((m, m), dtype=np.int64)
    for point in ref_points:
        idxs = sorted({id_to_idx[iid] for iid, _ in point.track if iid in id_to_idx})
        if len(idxs) >= 2:
            arr = np.asarray(idxs, dtype=np.intp)
            covis[np.ix_(arr, arr)] += 1
    np.fill_diagonal(covis, 0)

    neighbors: list[set[int]] = [set() for _ in range(m)]
    for i in range(m):
        row = covis[i]
        cand = np.where(row >= max(min_shared, 1))[0]
        if cand.size and top_k > 0:
            ranked = cand[np.argsort(row[cand])[::-1]]
            for j in ranked[:top_k]:
                neighbors[i].add(int(j))

    if geo_neighbors > 0 and len(ref_points) > 0:
        centroid = np.mean(np.asarray([p.xyz for p in ref_points], dtype=np.float64), axis=0)
        dirs = np.zeros((m, 3), dtype=np.float64)
        have = np.zeros(m, dtype=bool)
        for img in ref_images.values():
            if img.name in cam_index:
                i = cam_index[img.name]
                center = -img.rotmat.T @ img.tvec
                vec = center - centroid
                norm = float(np.linalg.norm(vec))
                dirs[i] = vec / norm if norm > 1e-9 else vec
                have[i] = True
        idx_have = np.where(have)[0]
        for i in idx_have:
            cos = dirs[idx_have] @ dirs[i]
            ranked = idx_have[np.argsort(cos)[::-1]]
            added = 0
            for j in ranked:
                if int(j) == int(i):
                    continue
                neighbors[i].add(int(j))
                added += 1
                if added >= geo_neighbors:
                    break

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(m):
        for j in neighbors[i]:
            a, b = (i, j) if i < j else (j, i)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b))
    out.sort()
    return [(names[a], names[b]) for a, b in out]


_MASK_CACHE: dict[str, "np.ndarray | None"] = {}


def load_camera_mask(mask_root: Path, image_name: str) -> "np.ndarray | None":
    """Per-camera keep/ignore mask (COLMAP convention: zero pixels = ignore keypoints).

    Looks for mask_root/<name>, mask_root/<name>.png, or mask_root/<stem>.<img ext>.
    Cached process-wide; a static rig has one mask per camera reused across frames."""
    cache_key = f"{mask_root}|{image_name}"
    if cache_key in _MASK_CACHE:
        return _MASK_CACHE[cache_key]
    import cv2

    stem = Path(image_name).stem
    candidates = [mask_root / image_name, mask_root / f"{image_name}.png"]
    candidates += [mask_root / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg", ".bmp")]
    mask = None
    for cand in candidates:
        if cand.exists():
            mask = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                LOGGER.warning("mask: failed to read %s; ignoring", cand)
            break
    _MASK_CACHE[cache_key] = mask
    return mask


def apply_feature_mask(feat: dict, mask: np.ndarray) -> dict:
    """Drop detected keypoints whose ORIGINAL-image pixel falls on a zero mask region."""
    kpts = feat["keypoints"][0]
    if kpts.shape[0] == 0:
        return feat
    import torch

    sx, sy = feat["scales"]
    resized_h, resized_w = feat["shape"]
    original_w = float(resized_w) * float(sx)
    original_h = float(resized_h) * float(sy)
    if original_w <= 0 or original_h <= 0 or mask.size == 0:
        raise ValueError("feature mask and image dimensions must be positive")
    xy = kpts.detach().cpu().numpy()
    # Masks may be stored at a lower resolution than the source image. Map through
    # original-image coordinates instead of clipping all out-of-range points onto
    # the mask's last row/column (which incorrectly kept or removed large regions).
    mask_sx = float(mask.shape[1]) / original_w
    mask_sy = float(mask.shape[0]) / original_h
    xs = np.clip(np.round(xy[:, 0] * sx * mask_sx).astype(np.int64), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(xy[:, 1] * sy * mask_sy).astype(np.int64), 0, mask.shape[0] - 1)
    keep_np = mask[ys, xs] > 0
    if keep_np.all():
        return feat
    keep = torch.from_numpy(keep_np).to(kpts.device)
    out = dict(feat)
    out["keypoints"] = [feat["keypoints"][0][keep]]
    out["scores"] = [feat["scores"][0][keep]]
    out["descriptors"] = [feat["descriptors"][0][:, keep]]
    return out


def process_resize(width: int, height: int, resize: Sequence[int]) -> tuple[int, int]:
    if len(resize) == 2:
        return int(resize[0]), int(resize[1])
    if len(resize) == 1 and resize[0] > 0:
        scale = float(resize[0]) / float(max(width, height))
        return int(round(width * scale)), int(round(height * scale))
    return width, height


class SuperGlueMatcher:
    def __init__(
        self,
        superglue_root: Path,
        weights: str,
        resize: Sequence[int],
        resize_float: bool,
        max_keypoints: int,
        keypoint_threshold: float,
        nms_radius: int,
        sinkhorn_iterations: int,
        match_threshold: float,
        device: str,
        fp16: bool = False,
        tf32: bool = True,
    ) -> None:
        self.superglue_root = superglue_root.resolve()
        if not self.superglue_root.exists():
            raise FileNotFoundError(f"SuperGlue root does not exist: {self.superglue_root}")
        sys.path.insert(0, str(self.superglue_root))

        import torch
        from models.matching import Matching

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; using CPU")
            device = "cpu"
        self.torch = torch
        self.device = device
        self.resize = list(resize)
        self.resize_float = resize_float
        # Camera input sizes repeat across frames, so let cuDNN autotune the best
        # conv kernels once. Deterministic for a given size -> output is unchanged.
        self.use_half = bool(fp16) and device == "cuda"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
        self.matching = Matching(
            {
                "superpoint": {
                    "nms_radius": nms_radius,
                    "keypoint_threshold": keypoint_threshold,
                    "max_keypoints": max_keypoints,
                },
                "superglue": {
                    "weights": weights,
                    "sinkhorn_iterations": sinkhorn_iterations,
                    "match_threshold": match_threshold,
                },
            }
        ).eval().to(device)
        self.torch.set_grad_enabled(False)
        self._placeholders: dict[tuple[int, int], object] = {}

    def _load_gray(self, path: Path, resize: Sequence[int] | None = None) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        width, height = image.shape[1], image.shape[0]
        new_width, new_height = process_resize(width, height, self.resize if resize is None else list(resize))
        scales = (float(width) / float(new_width), float(height) / float(new_height))
        if (new_width, new_height) != (width, height):
            image = cv2.resize(image.astype("float32" if self.resize_float else "uint8"), (new_width, new_height))
        return image, scales, (new_width, new_height)

    def _shape_placeholder(self, shape: tuple[int, int]):
        """A reusable (1, 1, H, W) tensor; SuperGlue only reads .shape for normalization."""
        key = (int(shape[0]), int(shape[1]))
        tensor = self._placeholders.get(key)
        if tensor is None:
            tensor = self.torch.empty((1, 1, key[0], key[1]), device=self.device)
            self._placeholders[key] = tensor
        return tensor

    def _autocast(self):
        """fp16 mixed precision for the heavy conv/matmul ops (Ampere/Ada speedup)."""
        if self.use_half:
            return self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
        return contextlib.nullcontext()

    def _decode(self, path: Path, resize: Sequence[int] | None = None) -> dict:
        """CPU-only image load + resize. Safe to run off the main thread (cv2 frees the GIL)."""
        img, scales, (new_width, new_height) = self._load_gray(path, resize)
        return {"img": img, "scales": scales, "shape": (new_height, new_width)}

    def _encode(self, decoded: dict) -> dict:
        """Run SuperPoint on a decoded image (GPU). Must be called on the CUDA thread."""
        tensor = self.torch.from_numpy(decoded["img"] / 255.0).float()[None, None].to(
            self.device, non_blocking=True
        )
        with self._autocast():
            pred = self.matching.superpoint({"image": tensor})
        # Keypoint coordinates reach ~4K and would lose precision in fp16 (exact ints
        # only up to 2048), so always keep them fp32 even when the network runs in fp16.
        keypoints = [k.float() for k in pred["keypoints"]]
        return {
            "keypoints": keypoints,              # list[Tensor(N, 2)] (resized-image coords)
            "scores": pred["scores"],
            "descriptors": pred["descriptors"],
            "scales": decoded["scales"],
            "shape": decoded["shape"],
        }

    def detect(self, path: Path, resize: Sequence[int] | None = None) -> dict:
        """Run SuperPoint once for an image; the result is reused across all its pairs."""
        return self._encode(self._decode(path, resize))

    def match_cached(self, feat0: dict, feat1: dict) -> np.ndarray:
        """Run only SuperGlue on two pre-detected feature sets."""
        data = {
            "image0": self._shape_placeholder(feat0["shape"]),
            "image1": self._shape_placeholder(feat1["shape"]),
            "keypoints0": feat0["keypoints"],
            "scores0": feat0["scores"],
            "descriptors0": feat0["descriptors"],
            "keypoints1": feat1["keypoints"],
            "scores1": feat1["scores"],
            "descriptors1": feat1["descriptors"],
        }
        with self._autocast():
            pred = self.matching(data)
        kpts0 = feat0["keypoints"][0].detach().cpu().numpy()
        kpts1 = feat1["keypoints"][0].detach().cpu().numpy()
        matches = pred["matches0"][0].detach().cpu().numpy()
        valid = matches > -1
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32)

        mkpts0 = kpts0[valid].astype(np.float32, copy=True)
        mkpts1 = kpts1[matches[valid]].astype(np.float32, copy=True)
        scales0 = feat0["scales"]
        scales1 = feat1["scales"]
        mkpts0[:, 0] *= scales0[0]
        mkpts0[:, 1] *= scales0[1]
        mkpts1[:, 0] *= scales1[0]
        mkpts1[:, 1] *= scales1[1]
        return np.concatenate([mkpts0, mkpts1], axis=1).astype(np.float32, copy=False)

    def match_pair(self, image0: Path, image1: Path, resize: Sequence[int] | None = None) -> np.ndarray:
        return self.match_cached(self.detect(image0, resize), self.detect(image1, resize))


class WaftFlowRunner:
    """In-process WAFT optical-flow inference (loads the model once, reuses it).

    Mirrors SuperGlueMatcher's lazy-import-from-an-external-repo pattern so the whole
    SuperGlue + COLMAP + WAFT pipeline runs from a single command/process.
    """

    def __init__(self, waft_root: Path, cfg: Path, ckpt: Path, device: str,
                 scale: float | None = None, max_size: int = 0) -> None:
        self.waft_root = Path(waft_root).resolve()
        if not self.waft_root.exists():
            raise FileNotFoundError(f"WAFT root does not exist: {self.waft_root}")
        cfg_path = cfg if Path(cfg).is_absolute() else self.waft_root / cfg
        ckpt_path = ckpt if Path(ckpt).is_absolute() else self.waft_root / ckpt
        if not cfg_path.exists():
            raise FileNotFoundError(f"WAFT config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"WAFT checkpoint not found: {ckpt_path}")
        sys.path.insert(0, str(self.waft_root))

        import torch
        from config.parser import json_to_args
        from model import fetch_model
        from utils.utils import load_ckpt
        from inference_tools import InferenceWrapper

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for WAFT but unavailable; using CPU")
            device = "cpu"
        self.torch = torch
        self.device = device
        self.max_size = int(max_size)

        cfg_args = json_to_args(str(cfg_path))
        if scale is not None:
            cfg_args.scale = scale
        model = fetch_model(cfg_args)
        load_ckpt(model, str(ckpt_path))
        model = model.to(device).eval()
        torch.set_grad_enabled(False)
        self.model = model
        self.wrapper = InferenceWrapper(
            model,
            scale=cfg_args.scale,
            train_size=getattr(cfg_args, "image_size", None),
            pad_to_train_size=False,
            tiling=False,
        )

    def _read_rgb(self, path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def flow(self, image0: Path, image1: Path) -> np.ndarray:
        """Forward optical flow image0 -> image1 as (H, W, 2) float32 in raw pixels.

        Output is at the input (native) resolution. If max_size > 0 the inputs are
        downscaled so the long side <= max_size and the resulting flow is rescaled back
        to that downscaled grid (still pixel-aligned to a same-ratio image)."""
        import cv2

        rgb0 = self._read_rgb(image0)
        rgb1 = self._read_rgb(image1)
        if rgb0.shape[:2] != rgb1.shape[:2]:
            raise RuntimeError(
                f"WAFT pair has mismatched sizes: {image0.name}{rgb0.shape[:2]} vs "
                f"{image1.name}{rgb1.shape[:2]}"
            )
        if self.max_size and max(rgb0.shape[:2]) > self.max_size:
            h, w = rgb0.shape[:2]
            s = self.max_size / float(max(h, w))
            new_wh = (max(1, int(round(w * s))), max(1, int(round(h * s))))
            rgb0 = cv2.resize(rgb0, new_wh, interpolation=cv2.INTER_AREA)
            rgb1 = cv2.resize(rgb1, new_wh, interpolation=cv2.INTER_AREA)

        t0 = self.torch.tensor(rgb0, dtype=self.torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t1 = self.torch.tensor(rgb1, dtype=self.torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        output = self.wrapper.calc_flow(t0, t1)
        flow = output["flow"][-1][0].permute(1, 2, 0).contiguous().cpu().numpy()
        return flow.astype(np.float32, copy=False)


def filter_corrs(
    corrs: np.ndarray,
    info0: ImageInfo,
    info1: ImageInfo,
    min_matches: int,
    ransac: bool,
    ransac_max_error: float,
    ransac_confidence: float,
    min_inlier_ratio: float = 0.0,
) -> np.ndarray:
    if corrs.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    corrs = np.asarray(corrs, dtype=np.float32).reshape(-1, 4)
    finite = np.isfinite(corrs).all(axis=1)
    in_bounds = (
        (corrs[:, 0] >= 0)
        & (corrs[:, 0] < info0.width)
        & (corrs[:, 1] >= 0)
        & (corrs[:, 1] < info0.height)
        & (corrs[:, 2] >= 0)
        & (corrs[:, 2] < info1.width)
        & (corrs[:, 3] >= 0)
        & (corrs[:, 3] < info1.height)
    )
    corrs = corrs[finite & in_bounds]
    if len(corrs) < min_matches:
        return np.empty((0, 4), dtype=np.float32)

    quant = np.round(corrs * 4.0).astype(np.int64)
    _, unique_idx = np.unique(quant, axis=0, return_index=True)
    corrs = corrs[np.sort(unique_idx)]

    if ransac and len(corrs) >= 8:
        import cv2

        before = len(corrs)
        _, mask = cv2.findFundamentalMat(
            corrs[:, :2],
            corrs[:, 2:],
            method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_max_error,
            confidence=ransac_confidence,
        )
        if mask is not None:
            corrs = corrs[mask.ravel() > 0]
            # A low geometric-inlier ratio is a hallmark of look-alike (but wrong)
            # image pairs; optionally reject the whole pair.
            if min_inlier_ratio > 0.0 and len(corrs) < min_inlier_ratio * before:
                return np.empty((0, 4), dtype=np.float32)
    if len(corrs) < min_matches:
        return np.empty((0, 4), dtype=np.float32)
    return corrs.astype(np.float32, copy=False)


def rotation_angle_deg(r: np.ndarray) -> float:
    cos = (float(np.trace(r)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def estimate_relative_rotation(
    corrs: np.ndarray,
    info0: ImageInfo,
    info1: ImageInfo,
    focal_factor: float,
    thresh_px: float = 2.0,
) -> tuple[np.ndarray | None, float]:
    """Rough relative rotation cam0->cam1 (x1 = R x0 + t) from the essential matrix,
    using the same focal guess the COLMAP database is initialised with.

    Returns (R, inlier_ratio); R is None when the pair is too weak to score."""
    import cv2

    corrs = np.asarray(corrs, dtype=np.float64).reshape(-1, 4)
    if len(corrs) < 15:
        return None, 0.0
    f0 = focal_factor * max(info0.width, info0.height)
    f1 = focal_factor * max(info1.width, info1.height)
    n0 = np.column_stack(((corrs[:, 0] - info0.width / 2.0) / f0, (corrs[:, 1] - info0.height / 2.0) / f0))
    n1 = np.column_stack(((corrs[:, 2] - info1.width / 2.0) / f1, (corrs[:, 3] - info1.height / 2.0) / f1))
    eye = np.eye(3, dtype=np.float64)
    e, mask = cv2.findEssentialMat(
        n0, n1, cameraMatrix=eye,
        method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
        prob=0.999, threshold=2.0 * thresh_px / (f0 + f1),
    )
    if e is None or mask is None:
        return None, 0.0
    if e.shape != (3, 3):
        e = e[:3, :3]
    inliers = mask.ravel() > 0
    ratio = float(inliers.sum()) / max(len(corrs), 1)
    if int(inliers.sum()) < 12:
        return None, ratio
    n_pose, r, _, _ = cv2.recoverPose(e, n0[inliers], n1[inliers], cameraMatrix=eye)
    if n_pose < 12:
        return None, ratio
    return np.asarray(r, dtype=np.float64), ratio


def cycle_filter_pairs(
    pair_matches: Sequence[PairMatch],
    image_infos: dict[str, ImageInfo],
    focal_factor: float,
    max_rot_error_deg: float,
    min_triangles: int,
    max_triangles_per_edge: int = 12,
) -> tuple[list[PairMatch], list[dict], int]:
    """Score every matched pair by rotation cycle consistency and drop outlier pairs.

    Look-alike cameras (symmetric rigs, repetitive texture) produce confident but WRONG
    pairs that individually pass RANSAC. A wrong pair however cannot compose: for a
    triangle (a,b,c) the rotations must satisfy R_ca @ R_bc @ R_ab ~ identity. A wrong
    pair corrupts EVERY triangle it participates in, so even its BEST (minimum-error)
    triangle is bad, while a good pair keeps at least one clean triangle. Pairs whose
    minimum cycle error exceeds the threshold are therefore dropped one at a time
    (worst first, rescoring after each drop so contaminated good pairs recover), and
    every camera keeps at least 2 pairs so the view graph stays usable."""
    rot: dict[tuple[str, str], np.ndarray] = {}
    ratio_by_pair: dict[tuple[str, str], float] = {}
    for pm in pair_matches:
        r, ratio = estimate_relative_rotation(pm.matches, image_infos[pm.name0], image_infos[pm.name1], focal_factor)
        ratio_by_pair[(pm.name0, pm.name1)] = ratio
        if r is not None:
            rot[(pm.name0, pm.name1)] = r
            rot[(pm.name1, pm.name0)] = r.T

    kept_keys: set[tuple[str, str]] = {(pm.name0, pm.name1) for pm in pair_matches}
    degree: dict[str, int] = {}
    for a, b in kept_keys:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    def score_edges() -> dict[tuple[str, str], tuple[float | None, int]]:
        """(min cycle error, #triangles) per kept edge with a known rotation."""
        adj: dict[str, set[str]] = {}
        for a, b in kept_keys:
            if (a, b) in rot:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        scores: dict[tuple[str, str], tuple[float | None, int]] = {}
        for a, b in kept_keys:
            if (a, b) not in rot:
                scores[(a, b)] = (None, 0)
                continue
            common = sorted(adj.get(a, set()) & adj.get(b, set()), key=natural_key)
            if len(common) > max_triangles_per_edge:
                idxs = np.linspace(0, len(common) - 1, max_triangles_per_edge).astype(int)
                common = [common[i] for i in idxs]
            errs = [
                rotation_angle_deg(rot[(c, a)] @ rot[(b, c)] @ rot[(a, b)])
                for c in common
            ]
            scores[(a, b)] = (min(errs) if errs else None, len(errs))
        return scores

    dropped: set[tuple[str, str]] = set()
    drop_score: dict[tuple[str, str], tuple[float, int]] = {}
    scores = score_edges()
    while True:
        candidates = sorted(
            ((err, key) for key, (err, n_tri) in scores.items()
             if err is not None and n_tri >= min_triangles and err > max_rot_error_deg),
            reverse=True,
        )
        removed = False
        for err, (a, b) in candidates:
            if degree[a] > 2 and degree[b] > 2:
                kept_keys.remove((a, b))
                dropped.add((a, b))
                drop_score[(a, b)] = (err, scores[(a, b)][1])
                degree[a] -= 1
                degree[b] -= 1
                removed = True
                break  # rescore: removing a bad edge cleans up its neighbours' triangles
        if not removed:
            break
        scores = score_edges()

    diagnostics: list[dict] = []
    for pm in pair_matches:
        key = (pm.name0, pm.name1)
        err, n_tri = drop_score.get(key) or scores.get(key, (None, 0))
        diagnostics.append({
            "name0": pm.name0,
            "name1": pm.name1,
            "matches": int(len(pm.matches)),
            "inlier_ratio": float(ratio_by_pair.get(key, 0.0)),
            "cycle_err_deg": round(err, 2) if err is not None else None,
            "triangles": n_tri,
            "dropped": key in dropped,
            "_corrs": pm.matches,
        })
    kept = [pm for pm in pair_matches if (pm.name0, pm.name1) not in dropped]
    return kept, diagnostics, len(dropped)


def render_pair_preview(
    path0: Path,
    path1: Path,
    corrs: np.ndarray,
    out_path: Path,
    label: str,
    dropped: bool,
    target_height: int = 720,
    max_lines: int = 80,
) -> None:
    import cv2

    img0 = cv2.imread(str(path0), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(path1), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        raise RuntimeError(f"failed to read preview images: {path0} / {path1}")
    s0 = target_height / img0.shape[0]
    s1 = target_height / img1.shape[0]
    img0 = cv2.resize(img0, (max(1, int(round(img0.shape[1] * s0))), target_height))
    img1 = cv2.resize(img1, (max(1, int(round(img1.shape[1] * s1))), target_height))
    canvas = np.concatenate([img0, img1], axis=1)
    corrs = np.asarray(corrs, dtype=np.float64).reshape(-1, 4)
    if len(corrs) > max_lines:
        corrs = corrs[np.linspace(0, len(corrs) - 1, max_lines).astype(int)]
    color = (0, 0, 255) if dropped else (0, 200, 0)
    off = img0.shape[1]
    for x0, y0, x1, y1 in corrs:
        p0 = (int(round(x0 * s0)), int(round(y0 * s0)))
        p1 = (int(round(x1 * s1)) + off, int(round(y1 * s1)))
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def write_pair_report(
    report_dir: Path,
    diagnostics: Sequence[dict],
    image_infos: dict[str, ImageInfo],
    max_previews: int,
    warn_err_deg: float,
) -> None:
    """Human-reviewable matching QC: a ranked table plus side-by-side previews of the
    dropped / most suspicious pairs, so the user can confirm them and extend
    --pair_blacklist (or fix --layout_file / masks) without rerunning blindly."""
    report_dir.mkdir(parents=True, exist_ok=True)

    def sort_key(d: dict):
        err = d["cycle_err_deg"]
        return (0 if d["dropped"] else 1, -(err if err is not None else -1.0))

    ranked = sorted(diagnostics, key=sort_key)
    lines = ["# name0 name1 matches inlier_ratio cycle_err_deg triangles status"]
    for d in ranked:
        err = "-" if d["cycle_err_deg"] is None else f"{d['cycle_err_deg']:.2f}"
        lines.append(
            f"{d['name0']} {d['name1']} {d['matches']} {d['inlier_ratio']:.3f} {err} "
            f"{d['triangles']} {'DROPPED' if d['dropped'] else 'kept'}"
        )
    (report_dir / "pairs_diagnostics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suspect_lines = [
        "# Pairs dropped by the rotation cycle-consistency filter.",
        "# Review the preview .jpg files, then paste confirmed lines into your --pair_blacklist file.",
    ]
    for d in ranked:
        if d["dropped"]:
            suspect_lines.append(
                f"{Path(d['name0']).stem} {Path(d['name1']).stem}  # cycle_err={d['cycle_err_deg']}deg"
            )
    (report_dir / "suspect_pairs.txt").write_text("\n".join(suspect_lines) + "\n", encoding="utf-8")

    previews = [
        d for d in ranked
        if d["dropped"] or (d["cycle_err_deg"] is not None and d["cycle_err_deg"] >= warn_err_deg)
    ][: max(0, max_previews)]
    for i, d in enumerate(previews):
        status = "dropped" if d["dropped"] else "kept"
        out = report_dir / f"{i:03d}_{Path(d['name0']).stem}__{Path(d['name1']).stem}_{status}.jpg"
        try:
            render_pair_preview(
                image_infos[d["name0"]].path, image_infos[d["name1"]].path, d["_corrs"], out,
                f"{d['name0']} | {d['name1']}  matches={d['matches']} inlier={d['inlier_ratio']:.2f} "
                f"cycle_err={d['cycle_err_deg']}deg [{status.upper()}]",
                dropped=d["dropped"],
            )
        except Exception as exc:
            LOGGER.warning("pair report: preview failed for %s-%s: %s", d["name0"], d["name1"], exc)
    n_dropped = sum(1 for d in diagnostics if d["dropped"])
    LOGGER.info(
        "pair report: %d dropped / %d pairs, %d preview image(s) -> %s",
        n_dropped, len(diagnostics), len(previews), report_dir,
    )


def index_pair_matches(
    image_infos: dict[str, ImageInfo],
    pair_matches: Sequence[PairMatch],
    keypoint_quantization: float,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, np.ndarray]]]:
    """Assign per-image keypoint indices to quantized match coordinates (vectorized).

    Coordinates within `keypoint_quantization` px collapse to one keypoint (first
    occurrence wins), same as the old per-point loop; only the keypoint ordering in
    the database differs, which COLMAP does not depend on."""
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in image_infos}
    lens: dict[str, int] = {name: 0 for name in image_infos}
    spans: list[tuple[int, int, int]] = []  # per pair: (n, offset0, offset1)
    for pair in pair_matches:
        m = np.asarray(pair.matches, dtype=np.float32).reshape(-1, 4)
        spans.append((len(m), lens[pair.name0], lens[pair.name1]))
        chunks[pair.name0].append(m[:, 0:2])
        lens[pair.name0] += len(m)
        chunks[pair.name1].append(m[:, 2:4])
        lens[pair.name1] += len(m)

    quant = max(float(keypoint_quantization), 1e-6)
    keypoints: dict[str, np.ndarray] = {}
    occurrence_idx: dict[str, np.ndarray] = {}
    for name, lst in chunks.items():
        if not lst:
            keypoints[name] = np.zeros((0, 2), dtype=np.float32)
            occurrence_idx[name] = np.zeros(0, dtype=np.int64)
            continue
        pts = np.concatenate(lst, axis=0)
        keys = np.round(pts.astype(np.float64) / quant).astype(np.int64)
        flat = keys[:, 0] * (1 << 32) + keys[:, 1]
        _, first_idx, inverse = np.unique(flat, return_index=True, return_inverse=True)
        keypoints[name] = pts[first_idx].astype(np.float32)
        occurrence_idx[name] = inverse.astype(np.int64)

    indexed_pairs: list[tuple[str, str, np.ndarray]] = []
    for pair, (n, off0, off1) in zip(pair_matches, spans):
        if n == 0:
            continue
        idx0 = occurrence_idx[pair.name0][off0:off0 + n]
        idx1 = occurrence_idx[pair.name1][off1:off1 + n]
        matches = np.unique(np.stack([idx0, idx1], axis=1).astype(np.uint32), axis=0)
        indexed_pairs.append((pair.name0, pair.name1, matches))
    return keypoints, indexed_pairs


def export_matches_to_database(
    database_path: Path,
    image_infos: dict[str, ImageInfo],
    pair_matches: Sequence[PairMatch],
    keypoint_quantization: float,
    two_view_config: int,
) -> dict[str, int]:
    keypoints, indexed_pairs = index_pair_matches(image_infos, pair_matches, keypoint_quantization)

    con = sqlite3.connect(database_path)
    try:
        con.execute("DELETE FROM keypoints")
        con.execute("DELETE FROM descriptors")
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM two_view_geometries")

        for name, arr in keypoints.items():
            image_id = image_infos[name].image_id
            con.execute(
                "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_id, int(arr.shape[0]), 2, array_to_blob(arr)),
            )

        pair_count = 0
        match_count = 0
        eye3 = array_to_blob(np.eye(3, dtype=np.float64))
        qvec = array_to_blob(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        tvec = array_to_blob(np.zeros(3, dtype=np.float64))
        for name0, name1, matches in indexed_pairs:
            info0 = image_infos[name0]
            info1 = image_infos[name1]
            if info0.image_id < info1.image_id:
                image_id0, image_id1 = info0.image_id, info1.image_id
                stored = matches
            else:
                image_id0, image_id1 = info1.image_id, info0.image_id
                stored = matches[:, ::-1].copy()
            pair_id = image_ids_to_pair_id(image_id0, image_id1)
            blob = array_to_blob(stored.astype(np.uint32, copy=False))
            rows = int(stored.shape[0])
            con.execute("INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)", (pair_id, rows, 2, blob))
            con.execute(
                "INSERT OR REPLACE INTO two_view_geometries"
                "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id, rows, 2, blob, two_view_config, eye3, eye3, eye3, qvec, tvec),
            )
            pair_count += 1
            match_count += rows
        con.commit()
        return {
            "images_with_keypoints": sum(1 for pts in keypoints.values() if len(pts)),
            "keypoints": sum(len(pts) for pts in keypoints.values()),
            "pairs": pair_count,
            "matches": match_count,
        }
    finally:
        con.close()


def append_matches_to_database(
    database_path: Path,
    image_infos: dict[str, ImageInfo],
    pair_matches: Sequence[PairMatch],
    keypoint_quantization: float,
    two_view_config: int,
) -> dict[str, int]:
    """Append new pairs without renumbering existing database keypoints.

    Preserving keypoint indices is essential when the database is used with an
    already reconstructed model: its POINT2D_IDX values refer to those indices.
    """
    if not pair_matches:
        return {"new_keypoints": 0, "pairs": 0, "matches": 0}
    quant = max(float(keypoint_quantization), 1e-6)
    used_names = {name for pair in pair_matches for name in (pair.name0, pair.name1)}
    con = sqlite3.connect(database_path)
    try:
        points_by_name: dict[str, list[tuple[float, float]]] = {}
        index_by_name: dict[str, dict[tuple[int, int], int]] = {}
        initial_counts: dict[str, int] = {}
        for name in used_names:
            info = image_infos[name]
            row = con.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (info.image_id,)
            ).fetchone()
            if row is None:
                array = np.empty((0, 2), dtype=np.float32)
            else:
                rows, cols, blob = int(row[0]), int(row[1]), row[2]
                array = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2].copy()
            points = [(float(x), float(y)) for x, y in array]
            indices: dict[tuple[int, int], int] = {}
            for index, (x, y) in enumerate(points):
                key = tuple(np.rint(np.asarray([x, y]) / quant).astype(np.int64))
                indices[key] = index
            points_by_name[name] = points
            index_by_name[name] = indices
            initial_counts[name] = len(points)

        indexed_pairs: list[tuple[str, str, np.ndarray]] = []
        for pair in pair_matches:
            indices = []
            for x0, y0, x1, y1 in np.asarray(pair.matches).reshape(-1, 4):
                pair_indices = []
                for name, x, y in ((pair.name0, x0, y0), (pair.name1, x1, y1)):
                    key = tuple(np.rint(np.asarray([x, y], dtype=np.float64) / quant).astype(np.int64))
                    index = index_by_name[name].get(key)
                    if index is None:
                        index = len(points_by_name[name])
                        index_by_name[name][key] = index
                        points_by_name[name].append((float(x), float(y)))
                    pair_indices.append(index)
                indices.append(tuple(pair_indices))
            matches = np.unique(np.asarray(indices, dtype=np.uint32).reshape(-1, 2), axis=0)
            if len(matches):
                indexed_pairs.append((pair.name0, pair.name1, matches))

        for name, points in points_by_name.items():
            array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            con.execute(
                "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_infos[name].image_id, int(array.shape[0]), 2, array_to_blob(array)),
            )

        eye3 = array_to_blob(np.eye(3, dtype=np.float64))
        qvec = array_to_blob(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        tvec = array_to_blob(np.zeros(3, dtype=np.float64))
        match_count = 0
        for name0, name1, matches in indexed_pairs:
            info0, info1 = image_infos[name0], image_infos[name1]
            if info0.image_id < info1.image_id:
                image_id0, image_id1, stored = info0.image_id, info1.image_id, matches
            else:
                image_id0, image_id1, stored = info1.image_id, info0.image_id, matches[:, ::-1].copy()
            pair_id = image_ids_to_pair_id(image_id0, image_id1)
            blob = array_to_blob(stored.astype(np.uint32, copy=False))
            rows = int(stored.shape[0])
            con.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, rows, 2, blob),
            )
            con.execute(
                "INSERT OR REPLACE INTO two_view_geometries"
                "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id, rows, 2, blob, two_view_config, eye3, eye3, eye3, qvec, tvec),
            )
            match_count += rows
        con.commit()
        return {
            "new_keypoints": sum(len(points_by_name[name]) - initial_counts[name] for name in used_names),
            "pairs": len(indexed_pairs),
            "matches": match_count,
        }
    finally:
        con.close()


def prepare_model_for_appended_keypoints(
    source_model_txt: Path,
    database_path: Path,
    output_model_txt: Path,
) -> Path:
    """Extend registered images with newly appended, untriangulated keypoints.

    COLMAP requires every registered image's POINTS2D array to have exactly the
    same length as its database keypoint row.  New entries are appended with a
    point3D id of -1, preserving all existing observation indices.
    """
    if output_model_txt.exists():
        shutil.rmtree(output_model_txt)
    output_model_txt.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_txt / "cameras.txt", output_model_txt / "cameras.txt")
    shutil.copy2(source_model_txt / "points3D.txt", output_model_txt / "points3D.txt")

    con = sqlite3.connect(database_path)
    try:
        keypoints_by_image_id: dict[int, np.ndarray] = {}
        for image_id, rows, cols, blob in con.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        ):
            keypoints_by_image_id[int(image_id)] = (
                np.frombuffer(blob, dtype=np.float32).reshape(int(rows), int(cols))[:, :2]
            )
    finally:
        con.close()

    source_lines = (source_model_txt / "images.txt").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    output_lines: list[str] = []
    index = 0
    while index < len(source_lines):
        header = source_lines[index]
        output_lines.append(header)
        index += 1
        stripped = header.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 10:
            raise ValueError(f"invalid COLMAP images.txt header: {header}")
        image_id = int(parts[0])
        points_line = source_lines[index] if index < len(source_lines) else ""
        index += 1
        point_tokens = points_line.split()
        if len(point_tokens) % 3:
            raise ValueError(f"invalid COLMAP POINTS2D line for image {image_id}")
        old_count = len(point_tokens) // 3
        database_points = keypoints_by_image_id.get(image_id)
        if database_points is None or len(database_points) < old_count:
            raise ValueError(
                f"database/model keypoint mismatch for image {image_id}: "
                f"database={0 if database_points is None else len(database_points)}, model={old_count}"
            )
        if len(database_points) > old_count:
            additions = " ".join(
                f"{float(x):.9g} {float(y):.9g} -1"
                for x, y in database_points[old_count:]
            )
            points_line = f"{points_line} {additions}".strip()
        output_lines.append(points_line)
    (output_model_txt / "images.txt").write_text(
        "\n".join(output_lines) + "\n", encoding="utf-8"
    )
    return output_model_txt


def model_num_registered_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if images_bin.exists():
        try:
            with images_bin.open("rb") as f:
                return int(np.frombuffer(f.read(8), dtype="<u8")[0])
        except Exception:
            return 0
    images_txt = model_dir / "images.txt"
    if images_txt.exists():
        count = 0
        for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
        return count // 2
    return 0


def resolve_init_pair(
    init_pair: Sequence[str] | None,
    image_infos: dict[str, ImageInfo],
    label: str,
) -> tuple[int, int] | None:
    """Map the user-provided --init_pair names (stem-tolerant) to database image ids."""
    if not init_pair:
        return None
    by_stem = {Path(n).stem: info for n, info in image_infos.items()}
    infos: list[ImageInfo] = []
    for token in init_pair:
        info = image_infos.get(token) or by_stem.get(Path(token).stem)
        if info is None:
            LOGGER.warning("%s: --init_pair image '%s' not found; ignoring the hint", label, token)
            return None
        infos.append(info)
    if infos[0].image_id == infos[1].image_id:
        LOGGER.warning("%s: --init_pair needs two different images; ignoring the hint", label)
        return None
    LOGGER.info("%s: mapper seeded with init pair (%s, %s)", label, infos[0].name, infos[1].name)
    return (infos[0].image_id, infos[1].image_id)


def run_mapper(
    colmap: str,
    database_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    log_dir: Path,
    args,
    init_ids: tuple[int, int] | None = None,
) -> Path:
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    def build_mapper_cmd(multiple_models: bool) -> list[str]:
        init_args = (
            ["--Mapper.init_image_id1", str(init_ids[0]), "--Mapper.init_image_id2", str(init_ids[1])]
            if init_ids is not None
            else []
        )
        return [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_refine_focal_length",
            "1" if args.ba_refine_focal_length else "0",
            "--Mapper.ba_refine_principal_point",
            "1" if args.ba_refine_principal_point else "0",
            "--Mapper.min_num_matches",
            str(args.mapper_min_num_matches),
            "--Mapper.init_min_num_inliers",
            str(args.mapper_init_min_num_inliers),
            "--Mapper.init_max_error",
            str(args.mapper_init_max_error),
            "--Mapper.abs_pose_min_num_inliers",
            str(args.mapper_abs_pose_min_num_inliers),
            "--Mapper.tri_min_angle",
            str(args.mapper_tri_min_angle),
            "--Mapper.multiple_models",
            "1" if multiple_models else "0",
        ] + init_args + (
            ["--Mapper.num_threads", str(args.mapper_num_threads)]
            if getattr(args, "mapper_num_threads", 0)
            else []
        )

    cmd = build_mapper_cmd(args.mapper_multiple_models)
    try:
        run_command(cmd, log_dir / "mapper.log")
    except CommandError as exc:
        message = str(exc)
        ba_single_image_crash = (
            "ba_config.NumImages() >= 2" in message
            or "At least two images must be registered for global bundle-adjustment" in message
        )
        if not (args.mapper_multiple_models and ba_single_image_crash):
            raise
        LOGGER.warning(
            "COLMAP mapper hit a multi-model global BA assertion; retrying as a single model "
            "(--Mapper.multiple_models 0)."
        )
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        run_command(build_mapper_cmd(False), log_dir / "mapper_retry_single_model.log")

    models = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError(f"COLMAP mapper produced no model in {sparse_dir}")
    best = max(models, key=model_num_registered_images)
    if len(models) > 1:
        sizes = {p.name: model_num_registered_images(p) for p in sorted(models, key=natural_key)}
        LOGGER.info("mapper produced %d models %s; selected '%s'", len(models), sizes, best.name)
    return best


def run_image_registrator(
    colmap: str,
    database_path: Path,
    input_model: Path,
    output_model: Path,
    log_dir: Path,
    args,
) -> Path:
    """Try to attach missing images to an existing, stable reconstruction."""
    if output_model.exists():
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "image_registrator",
        "--database_path",
        str(database_path),
        "--input_path",
        str(input_model),
        "--output_path",
        str(output_model),
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
        "--Mapper.abs_pose_min_num_inliers",
        str(args.mapper_abs_pose_min_num_inliers),
        "--Mapper.abs_pose_max_error",
        str(args.registration_rescue_abs_pose_max_error),
        "--Mapper.abs_pose_min_inlier_ratio",
        str(args.registration_rescue_abs_pose_min_inlier_ratio),
        "--Mapper.max_reg_trials",
        str(args.registration_rescue_max_trials),
        "--Mapper.ba_refine_focal_length",
        "1" if args.ba_refine_focal_length else "0",
        "--Mapper.ba_refine_principal_point",
        "1" if args.ba_refine_principal_point else "0",
        "--Mapper.fix_existing_frames",
        "1",
        "--Mapper.multiple_models",
        "0",
    ]
    if getattr(args, "mapper_num_threads", 0):
        cmd += ["--Mapper.num_threads", str(args.mapper_num_threads)]
    run_command(cmd, log_dir / "image_registrator_rescue.log")
    if not (output_model / "images.bin").exists() and not (output_model / "images.txt").exists():
        raise RuntimeError(f"COLMAP image_registrator produced no model in {output_model}")
    return output_model


def export_model_txt(colmap: str, model_dir: Path, txt_dir: Path, log_dir: Path) -> None:
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "model_converter", "--input_path", str(model_dir), "--output_path", str(txt_dir), "--output_type", "TXT"], log_dir / "model_converter_txt.log")


def run_dense(colmap: str, image_dir: Path, model_dir: Path, dense_dir: Path, fused_ply: Path, log_dir: Path, args) -> None:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "image_undistorter", "--image_path", str(image_dir), "--input_path", str(model_dir), "--output_path", str(dense_dir), "--output_type", "COLMAP"], log_dir / "image_undistorter.log")
    run_command([colmap, "patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.gpu_index", args.gpu_index], log_dir / "patch_match_stereo.log")
    run_command([colmap, "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(fused_ply), "--StereoFusion.check_num_images", str(args.fusion_check_num_images), "--StereoFusion.min_num_pixels", str(args.fusion_min_num_pixels)], log_dir / "stereo_fusion.log")


def parse_points3d_stats(points3d_txt: Path) -> dict[str, float | int | None]:
    if not points3d_txt.exists():
        return {"num_sparse_points": 0, "mean_reprojection_error": None, "median_reprojection_error": None}
    errors: list[float] = []
    count = 0
    for line in points3d_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            count += 1
            try:
                errors.append(float(parts[7]))
            except ValueError:
                pass
    return {
        "num_sparse_points": count,
        "mean_reprojection_error": float(np.mean(errors)) if errors else None,
        "median_reprojection_error": float(np.median(errors)) if errors else None,
    }


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    return np.asarray(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_colmap_text_model(model_txt_dir: Path) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    cameras: dict[int, ColmapCamera] = {}
    for line in (model_txt_dir / "cameras.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=np.asarray([float(x) for x in parts[4:]], dtype=np.float64),
        )

    images: dict[int, ColmapImage] = {}
    lines = (model_txt_dir / "images.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        points_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        values = points_line.split()
        triples = len(values) // 3
        xys = np.zeros((triples, 2), dtype=np.float64)
        point3d_ids = np.full(triples, -1, dtype=np.int64)
        for j in range(triples):
            xys[j, 0] = float(values[3 * j])
            xys[j, 1] = float(values[3 * j + 1])
            point3d_ids[j] = int(values[3 * j + 2])
        images[image_id] = ColmapImage(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3d_ids=point3d_ids,
            rotmat=qvec_to_rotmat(qvec),
        )

    points: list[ColmapPoint3D] = []
    for line in (model_txt_dir / "points3D.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        xyz = np.asarray([float(x) for x in parts[1:4]], dtype=np.float64)
        rgb = np.asarray([int(x) for x in parts[4:7]], dtype=np.uint8)
        error = float(parts[7])
        track_values = parts[8:]
        track: list[tuple[int, int]] = []
        for j in range(0, len(track_values) - 1, 2):
            track.append((int(track_values[j]), int(track_values[j + 1])))
        points.append(ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error, track=track))
    return cameras, images, points


def camera_normalized_from_pixel(camera: ColmapCamera, u: float, v: float) -> tuple[float, float]:
    p = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return (u - cx) / f, (v - cy) / f
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return (u - cx) / fx, (v - cy) / fy
    if model == "SIMPLE_RADIAL":
        import cv2

        f, cx, cy, k1 = p[:4]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, 0, 0, 0], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    if model == "RADIAL":
        import cv2

        f, cx, cy, k1, k2 = p[:5]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, k2, 0, 0], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    if model == "OPENCV":
        import cv2

        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        pts = np.asarray([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), np.asarray([k1, k2, p1, p2], dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    raise ValueError(f"unsupported camera model for velocity: {camera.model}")


def camera_project(camera: ColmapCamera, image: ColmapImage, xyz: np.ndarray) -> tuple[float, float, float]:
    xyz_cam = image.rotmat @ xyz + image.tvec
    depth = float(xyz_cam[2])
    if depth <= 1e-12:
        return np.nan, np.nan, depth
    x = float(xyz_cam[0] / depth)
    y = float(xyz_cam[1] / depth)
    p = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return f * x + cx, f * y + cy, depth
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return fx * x + cx, fy * y + cy, depth
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2
        return f * x * d + cx, f * y * d + cy, depth
    if model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        return f * x * d + cx, f * y * d + cy, depth
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return fx * xd + cx, fy * yd + cy, depth
    raise ValueError(f"unsupported camera model for velocity: {camera.model}")


def bilinear_sample_flow(flow: np.ndarray, u: float, v: float) -> tuple[float, float] | None:
    h, w = flow.shape[:2]
    if not (0 <= u <= w - 1 and 0 <= v <= h - 1):
        return None
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    wx = u - x0
    wy = v - y0
    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]
    value = (f00 * (1.0 - wx) + f01 * wx) * (1.0 - wy) + (f10 * (1.0 - wx) + f11 * wx) * wy
    if not np.isfinite(value).all():
        return None
    return float(value[0]), float(value[1])


def flow_to_pixel_delta(
    du: float,
    dv: float,
    flow_width: int,
    flow_height: int,
    image_width: int,
    image_height: int,
    flow_format: str,
    flow_scale: float,
) -> tuple[float, float]:
    if flow_format == "norm":
        return du * image_width * flow_scale, dv * image_height * flow_scale
    if flow_format == "midnorm":
        return du * image_width * 0.5 * flow_scale, dv * image_height * 0.5 * flow_scale
    return du * (image_width / flow_width) * flow_scale, dv * (image_height / flow_height) * flow_scale


def load_flow_maps(flow_dir: Path, image_names: Sequence[str]) -> dict[str, np.ndarray]:
    flows: dict[str, np.ndarray] = {}
    if not flow_dir.exists():
        return flows
    for name in image_names:
        path = flow_dir / f"{Path(name).stem}.npy"
        if not path.exists():
            continue
        try:
            arr = np.load(str(path))
        except Exception as exc:
            LOGGER.warning("failed to load flow %s: %s", path, exc)
            continue
        if arr.ndim == 3 and arr.shape[2] == 2:
            flows[name] = arr.astype(np.float32, copy=False)
    return flows


def triangulation_angle_deg(observations: Sequence[tuple[ColmapCamera, ColmapImage, float, float]]) -> float:
    rays = []
    for camera, image, u, v in observations:
        xn, yn = camera_normalized_from_pixel(camera, u, v)
        ray_cam = np.asarray([xn, yn, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = image.rotmat.T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        rays.append(ray_world)
    if len(rays) < 2:
        return 0.0
    max_angle = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            dot = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            max_angle = max(max_angle, float(np.degrees(np.arccos(dot))))
    return max_angle


def triangulate_observations(observations: Sequence[tuple[ColmapCamera, ColmapImage, float, float]]) -> np.ndarray | None:
    rows = []
    for camera, image, u, v in observations:
        xn, yn = camera_normalized_from_pixel(camera, u, v)
        rt = np.hstack([image.rotmat, image.tvec.reshape(3, 1)])
        rows.append(xn * rt[2] - rt[0])
        rows.append(yn * rt[2] - rt[1])
    if len(rows) < 4:
        return None
    a = np.asarray(rows, dtype=np.float64)
    scale = np.linalg.norm(a, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    a = a / scale
    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1]
    if abs(float(h[3])) < 1e-12:
        return None
    xyz = h[:3] / h[3]
    if not np.isfinite(xyz).all():
        return None
    return xyz


VELOCITY_DROP_KEYS = ("too_few_views", "triangulate", "gate", "depth", "reproj", "angle", "magnitude")


def velocity_from_observations(
    point_xyz: np.ndarray,
    observations: list,
    args,
    track_len: int,
    velocity_dt_multiplier: float = 1.0,
):
    """Triangulate flow-displaced observations into a 3D velocity for ONE point.

    observations: list[(ColmapCamera, ColmapImage, up, vp)] (already flow-displaced).
    Returns (result, drop_reason). On success result = (velocity(3,) float32, confidence,
    view_count) and drop_reason is None; on rejection result is None and drop_reason names
    the gate (one of VELOCITY_DROP_KEYS)."""
    min_views = args.velocity_min_views
    if len(observations) < min_views:
        return None, "too_few_views"
    gate_schedule = [
        args.velocity_max_reproj_error * 4.0,
        args.velocity_max_reproj_error * 2.0,
        args.velocity_max_reproj_error,
    ]
    active = observations
    xyz_next = triangulate_observations(active)
    if xyz_next is None:
        return None, "triangulate"
    for gate in gate_schedule:
        gated = []
        for obs in active:
            camera, image, up, vp = obs
            pu, pv, depth = camera_project(camera, image, xyz_next)
            resid = float(np.hypot(pu - up, pv - vp)) if np.isfinite(pu) and np.isfinite(pv) else np.inf
            if depth > 1e-8 and resid <= gate:
                gated.append(obs)
        if len(gated) < min_views:
            return None, "gate"
        active = gated
        xyz_refined = triangulate_observations(active)
        if xyz_refined is None:
            return None, "gate"
        xyz_next = xyz_refined

    final_residuals = []
    positive_depths = 0
    for camera, image, up, vp in active:
        pu, pv, depth = camera_project(camera, image, xyz_next)
        if depth > 1e-8:
            positive_depths += 1
        final_residuals.append(float(np.hypot(pu - up, pv - vp)) if np.isfinite(pu) and np.isfinite(pv) else np.inf)
    if positive_depths < min_views:
        return None, "depth"
    median_resid = float(np.median(final_residuals))
    if median_resid > args.velocity_max_reproj_error:
        return None, "reproj"
    angle = triangulation_angle_deg(active)
    if angle < args.velocity_min_angle:
        return None, "angle"

    velocity = (xyz_next - point_xyz) / (args.velocity_dt * velocity_dt_multiplier)
    speed = float(np.linalg.norm(velocity))
    if args.velocity_max_magnitude > 0 and speed > args.velocity_max_magnitude:
        return None, "magnitude"

    track_ratio = len(active) / max(track_len, 1)
    reproj_score = np.exp(-median_resid / max(args.velocity_max_reproj_error, 1e-6))
    angle_score = min(angle / max(args.velocity_min_angle * 4.0, 1e-6), 1.0)
    confidence = float(np.clip(track_ratio * reproj_score * angle_score, 0.0, 1.0))
    return (velocity.astype(np.float32), confidence, len(active)), None


def log_velocity_summary(label: str, valid_count: int, n: int, drops: dict,
                         applied_disp_px: list, median_width: float, args) -> None:
    LOGGER.info("%s: %d/%d points passed multi-view flow triangulation", label, valid_count, n)
    LOGGER.info(
        "%s drops: too_few_views=%d triangulate=%d gate=%d depth=%d reproj=%d angle=%d magnitude=%d",
        label, drops["too_few_views"], drops["triangulate"], drops["gate"], drops["depth"],
        drops["reproj"], drops["angle"], drops["magnitude"],
    )
    if applied_disp_px:
        median_disp = float(np.median(applied_disp_px))
        LOGGER.info(
            "%s: median applied flow displacement = %.3f px (flow_format=%s, flow_direction=%s)",
            label, median_disp, args.flow_format, getattr(args, "flow_direction", "forward"),
        )
        if median_width > 0 and median_disp > 0.1 * median_width:
            LOGGER.warning(
                "%s: median flow displacement %.1f px is >10%% of image width (%.0f px) — this usually "
                "means --flow_format is wrong for your flow files (WAFT outputs raw pixels -> --flow_format "
                "pixel) or the flow was measured in a different image space than the model.",
                label, median_disp, median_width,
            )


def estimate_flow_velocity_for_points(
    points: Sequence[ColmapPoint3D],
    cameras: dict[int, ColmapCamera],
    images: dict[int, ColmapImage],
    flow_dir: Path,
    args,
    velocity_dt_multiplier: float = 1.0,
) -> VelocityResult:
    n = len(points)
    velocities = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float32)
    view_counts = np.zeros(n, dtype=np.uint8)
    flows = load_flow_maps(flow_dir, [img.name for img in images.values()])
    if not flows:
        LOGGER.info("velocity: no flow files found in %s", flow_dir)
        return VelocityResult(velocities, valid, confidence, view_counts)

    direction_sign = -1.0 if getattr(args, "flow_direction", "forward") == "backward" else 1.0
    drops = {k: 0 for k in VELOCITY_DROP_KEYS}
    applied_disp_px: list[float] = []
    for idx, point in enumerate(points):
        observations: list[tuple[ColmapCamera, ColmapImage, float, float]] = []
        for image_id, point2d_idx in point.track:
            image = images.get(image_id)
            if image is None or point2d_idx < 0 or point2d_idx >= len(image.xys):
                continue
            camera = cameras[image.camera_id]
            flow = flows.get(image.name)
            if flow is None:
                continue
            u, v = image.xys[point2d_idx]
            if not (0 <= u < camera.width and 0 <= v < camera.height):
                continue
            flow_h, flow_w = flow.shape[:2]
            sample = bilinear_sample_flow(flow, u * flow_w / camera.width, v * flow_h / camera.height)
            if sample is None:
                continue
            du, dv = flow_to_pixel_delta(
                sample[0], sample[1], flow_w, flow_h, camera.width, camera.height,
                args.flow_format, args.flow_scale,
            )
            du *= direction_sign
            dv *= direction_sign
            applied_disp_px.append(float(np.hypot(du, dv)))
            up = float(u + du)
            vp = float(v + dv)
            if 0 <= up < camera.width and 0 <= vp < camera.height:
                observations.append((camera, image, up, vp))

        result, reason = velocity_from_observations(
            point.xyz, observations, args, len(point.track), velocity_dt_multiplier
        )
        if reason is not None:
            drops[reason] += 1
            continue
        velocity, conf, views = result
        velocities[idx] = velocity
        valid[idx] = True
        view_counts[idx] = min(views, 255)
        confidence[idx] = conf

    median_w = float(np.median([c.width for c in cameras.values()])) if cameras else 0.0
    log_velocity_summary("velocity", int(valid.sum()), n, drops, applied_disp_px, median_w, args)
    return VelocityResult(velocities, valid, confidence, view_counts)


def estimate_velocity_undistorted(
    points_xyz: np.ndarray,
    points_cam_names: list,
    undist_cam_by_name: dict,
    rig_image_by_name: dict,
    flow_dir: Path,
    args,
    velocity_dt_multiplier: float = 1.0,
) -> VelocityResult:
    """Velocity in the UNDISTORTED PINHOLE space, from flow measured on undistorted images.

    Each 3D point is projected through the shared PINHOLE camera (rig poses are identical
    across frames), the undistorted-space flow is sampled there, the point is displaced and
    re-triangulated. Flow is native-resolution so sampling is exact."""
    n = len(points_xyz)
    velocities = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float32)
    view_counts = np.zeros(n, dtype=np.uint8)
    needed = sorted({nm for lst in points_cam_names for nm in lst}, key=natural_key)
    flows = load_flow_maps(flow_dir, needed)
    if not flows:
        LOGGER.info("velocity(undist): no flow files found in %s", flow_dir)
        return VelocityResult(velocities, valid, confidence, view_counts)

    direction_sign = -1.0 if getattr(args, "flow_direction", "forward") == "backward" else 1.0
    drops = {k: 0 for k in VELOCITY_DROP_KEYS}
    applied_disp_px: list[float] = []
    for idx in range(n):
        xyz = np.asarray(points_xyz[idx], dtype=np.float64)
        observations: list[tuple[ColmapCamera, ColmapImage, float, float]] = []
        for name in points_cam_names[idx]:
            camera = undist_cam_by_name.get(name)
            image = rig_image_by_name.get(name)
            flow = flows.get(name)
            if camera is None or image is None or flow is None:
                continue
            pu, pv, depth = camera_project(camera, image, xyz)
            if depth <= 1e-8 or not np.isfinite(pu) or not np.isfinite(pv):
                continue
            if not (0 <= pu < camera.width and 0 <= pv < camera.height):
                continue
            flow_h, flow_w = flow.shape[:2]
            sample = bilinear_sample_flow(flow, pu * flow_w / camera.width, pv * flow_h / camera.height)
            if sample is None:
                continue
            du, dv = flow_to_pixel_delta(
                sample[0], sample[1], flow_w, flow_h, camera.width, camera.height,
                args.flow_format, args.flow_scale,
            )
            du *= direction_sign
            dv *= direction_sign
            applied_disp_px.append(float(np.hypot(du, dv)))
            up = float(pu + du)
            vp = float(pv + dv)
            if 0 <= up < camera.width and 0 <= vp < camera.height:
                observations.append((camera, image, up, vp))

        result, reason = velocity_from_observations(
            xyz, observations, args, len(points_cam_names[idx]), velocity_dt_multiplier
        )
        if reason is not None:
            drops[reason] += 1
            continue
        velocity, conf, views = result
        velocities[idx] = velocity
        valid[idx] = True
        view_counts[idx] = min(views, 255)
        confidence[idx] = conf

    median_w = float(np.median([c.width for c in undist_cam_by_name.values()])) if undist_cam_by_name else 0.0
    log_velocity_summary("velocity(undist)", int(valid.sum()), n, drops, applied_disp_px, median_w, args)
    return VelocityResult(velocities, valid, confidence, view_counts)


def write_point_cache(path: Path, points: Sequence[ColmapPoint3D], images: dict[int, ColmapImage]) -> None:
    """Persist (xyz, rgb, per-point observing camera names) for the deferred velocity pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    xyz = np.array([p.xyz for p in points], dtype=np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32)
    rgb = np.array([p.rgb for p in points], dtype=np.uint8).reshape(n, 3) if n else np.zeros((0, 3), np.uint8)
    name_lists: list[list[str]] = []
    for p in points:
        names = [images[i].name for i, _ in p.track if i in images]
        name_lists.append(names)
    table = sorted({nm for names in name_lists for nm in names}, key=natural_key)
    table_index = {nm: i for i, nm in enumerate(table)}
    offsets = np.zeros(n + 1, dtype=np.int64)
    flat: list[int] = []
    for i, names in enumerate(name_lists):
        flat.extend(table_index[nm] for nm in names)
        offsets[i + 1] = len(flat)
    np.savez(
        path,
        xyz=xyz,
        rgb=rgb,
        cam_table=np.array(table, dtype="U") if table else np.array([], dtype="U1"),
        idx_flat=np.array(flat, dtype=np.int64),
        offsets=offsets,
    )


def read_point_cache(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    data = np.load(path, allow_pickle=False)
    xyz = data["xyz"]
    rgb = data["rgb"]
    cam_table = [str(x) for x in data["cam_table"]]
    idx_flat = data["idx_flat"]
    offsets = data["offsets"]
    name_lists = [
        [cam_table[j] for j in idx_flat[offsets[i]:offsets[i + 1]]]
        for i in range(len(xyz))
    ]
    return xyz, rgb, name_lists


def build_flow_frame_pairs(selected_frames: Sequence[Path], frame_interval: int) -> list[tuple[str, str]]:
    """Return source/target frame names for the configured flow anchors.

    interval=1 preserves the old adjacent-frame behavior. interval=5 uses the
    1st, 5th, 10th, ... selected frames as anchors, matching the requested
    "first, fifth, tenth" sampling convention.
    """
    frame_names = [f.name for f in selected_frames]
    if len(frame_names) < 2:
        return []
    if frame_interval <= 1:
        anchor_indices = list(range(len(frame_names)))
    else:
        anchor_indices = [0]
        anchor_indices.extend(i for i in range(frame_interval - 1, len(frame_names), frame_interval) if i != 0)
    return [
        (frame_names[anchor_indices[i]], frame_names[anchor_indices[i + 1]])
        for i in range(len(anchor_indices) - 1)
    ]


def write_velocity_npz(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    velocity: VelocityResult,
    source_frame: str,
    target_frame: str | None,
    frame_interval: int,
    velocity_dt: float,
) -> None:
    """Write velocities aligned 1:1 with the corresponding points/<frame>.ply vertices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=np.asarray(xyz, dtype=np.float32),
        rgb=np.asarray(rgb, dtype=np.uint8),
        velocity=velocity.velocities.astype(np.float32, copy=False),
        valid=velocity.valid.astype(np.bool_, copy=False),
        confidence=velocity.confidence.astype(np.float32, copy=False),
        view_counts=velocity.view_counts.astype(np.uint8, copy=False),
        source_frame=np.array(source_frame),
        target_frame=np.array("" if target_frame is None else target_frame),
        frame_interval=np.array(int(frame_interval), dtype=np.int32),
        velocity_dt=np.array(float(velocity_dt), dtype=np.float32),
    )


def compute_flows_for_frames(selected_frames: Sequence[Path], args, runner: WaftFlowRunner) -> int:
    """Phase 2: forward WAFT flow between configured frame anchors, per camera.

    Writes output_root/flows/<source>/<cam>.npy aligned 1:1 with undistorted/images/<source>/<cam>."""
    images_root = args.output_root / "undistorted" / "images"
    flows_root = args.output_root / "flows"
    frame_pairs = build_flow_frame_pairs(selected_frames, args.flow_frame_interval)
    if not frame_pairs:
        LOGGER.warning(
            "flow: no frame pairs for --flow_frame_interval=%d over %d selected frames",
            args.flow_frame_interval, len(selected_frames),
        )
        return 0

    # Build the work list once so the progress bar has an exact total.
    pairs: list[tuple[str, str, list[str], dict, dict]] = []
    for a, b in frame_pairs:
        dir_a = images_root / a
        dir_b = images_root / b
        if not dir_a.exists() or not dir_b.exists():
            LOGGER.warning("flow: missing undistorted images for %s or %s; skipping pair", a, b)
            continue
        files_a = {p.stem: p for p in dir_a.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        files_b = {p.stem: p for p in dir_b.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        common = sorted(set(files_a) & set(files_b), key=natural_key)
        pairs.append((a, b, common, files_a, files_b))

    grand_total = sum(len(c) for _, _, c, _, _ in pairs)
    LOGGER.info("flow: computing %d flow maps over %d frame pairs", grand_total, len(pairs))
    bar = make_pbar(grand_total, "flow", "map", args.progress)
    total = 0
    for a, b, common, files_a, files_b in pairs:
        out_dir = flows_root / a
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "_pair.json"
        skip_existing_for_pair = False
        if args.skip_existing_flow and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                skip_existing_for_pair = (
                    meta.get("source_frame") == a
                    and meta.get("target_frame") == b
                    and int(meta.get("flow_frame_interval", 1)) == int(args.flow_frame_interval)
                )
            except Exception as exc:
                LOGGER.warning("flow: failed to read %s for resume check: %s", meta_path, exc)
        wrote = 0
        bar.set_postfix_str(f"{a}->{b}")
        for cam in common:
            out_path = out_dir / f"{cam}.npy"
            if not (skip_existing_for_pair and out_path.exists()):
                flow = runner.flow(files_a[cam], files_b[cam])
                np.save(out_path, flow)
                wrote += 1
                total += 1
            bar.update(1)
        meta_path.write_text(
            json.dumps(
                {
                    "source_frame": a,
                    "target_frame": b,
                    "flow_frame_interval": int(args.flow_frame_interval),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("flow: %s->%s wrote %d/%d cameras", a, b, wrote, len(common))
    bar.close()
    LOGGER.info("flow: wrote %d flow maps under %s", total, flows_root)
    return total


def run_undistorted_velocity_pass(args, colmap: str, selected_frames: Sequence[Path]) -> None:
    """Phase 3: compute vx/vy/vz for frames that have configured forward flow."""
    shared_model = args.output_root / "undistorted" / "sparse" / "0"
    if not shared_model.exists():
        LOGGER.warning("velocity: shared undistorted model %s missing; skipping velocity pass", shared_model)
        return
    cache_dir = args.output_root / "_flowcache"
    cameras, images, _ = read_colmap_model_any(colmap, shared_model, cache_dir / "_shared_txt", cache_dir)
    undist_cam_by_name = {img.name: cameras[img.camera_id] for img in images.values()}
    rig_image_by_name = {img.name: img for img in images.values()}
    flows_root = args.output_root / "flows"
    frame_pairs = build_flow_frame_pairs(selected_frames, args.flow_frame_interval)
    for name, target_name in frame_pairs:
        cache_path = cache_dir / f"{name}.npz"
        out_ply = args.output_root / "points" / f"{name}.ply"
        out_velocity = args.output_root / "velocities" / f"{name}.npz"
        if not cache_path.exists():
            LOGGER.warning("velocity: point cache %s missing; skipping %s", cache_path, name)
            continue
        xyz, rgb, name_lists = read_point_cache(cache_path)
        flow_dir = flows_root / name
        if not flow_dir.exists() or not any(flow_dir.glob("*.npy")):
            LOGGER.warning("velocity: %s has no flow files for %s->%s; skipping", name, name, target_name)
            continue
        velocity_result = estimate_velocity_undistorted(
            xyz, name_lists, undist_cam_by_name, rig_image_by_name, flow_dir, args, args.flow_frame_interval
        )
        write_velocity_npz(
            out_velocity, xyz, rgb, velocity_result, name, target_name,
            args.flow_frame_interval, args.velocity_dt,
        )
        pts = [
            ColmapPoint3D(point_id=i + 1, xyz=xyz[i], rgb=rgb[i], error=0.0, track=[])
            for i in range(len(xyz))
        ]
        write_points_ply(out_ply, pts, velocity_result, ascii=args.ply_ascii)
        LOGGER.info("velocity: wrote %s and updated %s", out_velocity, out_ply)
    if not args.keep_workspace:
        shutil.rmtree(cache_dir, ignore_errors=True)


def filter_points(
    points: Sequence[ColmapPoint3D],
    min_track_len: int,
    max_reproj_error: float,
) -> list[ColmapPoint3D]:
    """Drop noisy sparse points: short tracks and high-reprojection-error points.

    A value of 0 disables the corresponding criterion.
    """
    if min_track_len <= 0 and max_reproj_error <= 0:
        return list(points)
    kept: list[ColmapPoint3D] = []
    for point in points:
        if min_track_len > 0 and len(point.track) < min_track_len:
            continue
        if max_reproj_error > 0 and point.error > max_reproj_error:
            continue
        kept.append(point)
    return kept


def write_points_ply(
    ply_path: Path,
    points: Sequence[ColmapPoint3D],
    velocity: VelocityResult | None,
    ascii: bool = True,
) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    if velocity is None:
        velocity = VelocityResult(
            velocities=np.zeros((n, 3), dtype=np.float32),
            valid=np.zeros(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float32),
            view_counts=np.zeros(n, dtype=np.uint8),
        )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("vx", "<f4"),
            ("vy", "<f4"),
            ("vz", "<f4"),
            ("velocity_confidence", "<f4"),
            ("velocity_valid", "u1"),
            ("velocity_views", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    if n:
        xyz = np.asarray([p.xyz for p in points], dtype=np.float32)
        rgb = np.asarray([p.rgb for p in points], dtype=np.uint8)
        data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        data["vx"], data["vy"], data["vz"] = velocity.velocities[:, 0], velocity.velocities[:, 1], velocity.velocities[:, 2]
        data["velocity_confidence"] = velocity.confidence.astype(np.float32, copy=False)
        data["velocity_valid"] = velocity.valid.astype(np.uint8)
        data["velocity_views"] = velocity.view_counts.astype(np.uint8, copy=False)
    fmt = "ascii 1.0" if ascii else "binary_little_endian 1.0"
    header = (
        "ply\n"
        f"format {fmt}\n"
        "comment velocity is estimated by multi-view optical-flow triangulation\n"
        "comment vx vy vz are in COLMAP world units per velocity_dt\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float vx\n"
        "property float vy\n"
        "property float vz\n"
        "property float velocity_confidence\n"
        "property uchar velocity_valid\n"
        "property uchar velocity_views\n"
        "end_header\n"
    )
    with ply_path.open("wb") as f:
        f.write(header.encode("ascii"))
        if not ascii:
            f.write(data.tobytes())
        elif n:
            # One human-readable line per vertex. %.9g round-trips float32 exactly;
            # color/flag columns print as integers.
            row_fmt = "%.9g %.9g %.9g %d %d %d %.9g %.9g %.9g %.9g %d %d"
            np.savetxt(
                f,
                np.array(data.tolist(), dtype=object),
                fmt=row_fmt,
            )


def copy_sparse_model(model_dir: Path, output_root: Path, output_name: str, force: bool) -> Path:
    final_dir = output_root / "sparse" / output_name / model_dir.name
    if final_dir.exists():
        if force:
            shutil.rmtree(final_dir)
        else:
            raise FileExistsError(f"sparse output already exists: {final_dir}. Use --force to overwrite.")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, final_dir)
    return final_dir


def camera_focal(camera: ColmapCamera) -> float:
    return float(camera.params[0])


def pixels_to_normalized(pts: np.ndarray, camera: ColmapCamera) -> np.ndarray:
    """Convert pixel coordinates to undistorted normalized camera coordinates (batched)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return np.column_stack(((pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f))
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return np.column_stack(((pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy))

    import cv2

    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        k = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray([k1, 0, 0, 0], dtype=np.float64)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        k = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray([k1, k2, 0, 0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray([k1, k2, p1, p2], dtype=np.float64)
    else:
        raise ValueError(f"unsupported camera model for epipolar filter: {model}")
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), k, dist)
    return und.reshape(-1, 2)


def essential_matrix(img0: ColmapImage, img1: ColmapImage) -> np.ndarray:
    """Essential matrix E with x1^T E x0 = 0 for normalized coords, from world->cam poses."""
    r_rel = img1.rotmat @ img0.rotmat.T
    t_rel = img1.tvec - r_rel @ img0.tvec
    tx = np.asarray(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ],
        dtype=np.float64,
    )
    return tx @ r_rel


def epipolar_filter_corrs(
    corrs: np.ndarray,
    cam0: ColmapCamera,
    img0: ColmapImage,
    cam1: ColmapCamera,
    img1: ColmapImage,
    max_error_px: float,
) -> np.ndarray:
    """Drop matches that violate the epipolar constraint of the known relative pose."""
    corrs = np.asarray(corrs, dtype=np.float64).reshape(-1, 4)
    if corrs.size == 0:
        return corrs.astype(np.float32, copy=False)
    n0 = pixels_to_normalized(corrs[:, 0:2], cam0)
    n1 = pixels_to_normalized(corrs[:, 2:4], cam1)
    e = essential_matrix(img0, img1)
    x0 = np.column_stack((n0, np.ones(len(n0))))
    x1 = np.column_stack((n1, np.ones(len(n1))))
    ex0 = x0 @ e.T          # each row = E x0_i
    etx1 = x1 @ e           # each row = E^T x1_i
    num = np.sum(x1 * ex0, axis=1) ** 2
    denom = ex0[:, 0] ** 2 + ex0[:, 1] ** 2 + etx1[:, 0] ** 2 + etx1[:, 1] ** 2
    denom = np.where(denom < 1e-12, 1e-12, denom)
    err_px = np.sqrt(num / denom) * (0.5 * (camera_focal(cam0) + camera_focal(cam1)))
    return corrs[err_px <= max_error_px].astype(np.float32, copy=False)


def epipolar_filter_pairs(
    pair_matches: Sequence[PairMatch],
    cameras_by_id: dict[int, ColmapCamera],
    images_by_name: dict[str, ColmapImage],
    max_error_px: float,
    min_matches: int,
) -> tuple[list[PairMatch], int, int]:
    """Epipolar-clean every pair using known poses. Pairs without a known pose pass through."""
    filtered: list[PairMatch] = []
    n_in = 0
    n_out = 0
    for pm in pair_matches:
        n_in += len(pm.matches)
        img0 = images_by_name.get(pm.name0)
        img1 = images_by_name.get(pm.name1)
        if img0 is None or img1 is None:
            filtered.append(pm)
            n_out += len(pm.matches)
            continue
        kept = epipolar_filter_corrs(pm.matches, cameras_by_id[img0.camera_id], img0, cameras_by_id[img1.camera_id], img1, max_error_px)
        n_out += len(kept)
        if len(kept) >= min_matches:
            filtered.append(PairMatch(pm.name0, pm.name1, kept))
    return filtered, n_in, n_out


def run_undistort(colmap: str, image_dir: Path, model_dir: Path, undistort_dir: Path, log_dir: Path, args) -> None:
    """Write undistorted images + a distortion-free (PINHOLE) model via COLMAP image_undistorter."""
    if undistort_dir.exists():
        shutil.rmtree(undistort_dir)
    undistort_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(model_dir),
        "--output_path",
        str(undistort_dir),
        "--output_type",
        "COLMAP",
    ]
    if args.undistort_max_image_size and args.undistort_max_image_size > 0:
        cmd += ["--max_image_size", str(args.undistort_max_image_size)]
    run_command(cmd, log_dir / "image_undistorter.log")


@dataclass
class ExportContext:
    """Holds the shared 3DGS export targets and the per-camera centering crops.

    In --static_rig the crops + shared sparse model are computed once on the reference
    frame and reused, so every frame writes images that match one shared camera set.
    """
    images_root: Path        # output_root/undistorted/images
    sparse_dir: Path         # output_root/undistorted/sparse/0
    crop_by_name: dict[str, tuple[int, int, int, int]] | None = None


def center_crop_spec(width: int, height: int, cx: float, cy: float) -> tuple[int, int, int, int]:
    """Largest centered (left, top, w, h) window whose centre is the principal point."""
    cxr = int(round(cx))
    cyr = int(round(cy))
    hw = max(1, min(cxr, width - cxr))
    hh = max(1, min(cyr, height - cyr))
    return cxr - hw, cyr - hh, 2 * hw, 2 * hh


def crop_image_file(src: Path, dst: Path, spec: tuple[int, int, int, int]) -> None:
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read undistorted image: {src}")
    left, top, w, h = spec
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img[top:top + h, left:left + w])


def read_colmap_model_any(colmap: str, model_dir: Path, txt_tmp: Path, log_dir: Path):
    """Convert a (binary) COLMAP model to text and read it."""
    if txt_tmp.exists():
        shutil.rmtree(txt_tmp)
    txt_tmp.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "model_converter", "--input_path", str(model_dir), "--output_path", str(txt_tmp), "--output_type", "TXT"], log_dir / "undist_model_converter.log")
    return read_colmap_text_model(txt_tmp)


def write_colmap_text_model(
    tmp_dir: Path,
    cameras: dict[int, ColmapCamera],
    images: dict[int, ColmapImage],
    points: Sequence[ColmapPoint3D],
    crops: dict[int, tuple[int, int, int, int]],
) -> None:
    """Write cameras/images/points3D .txt, shifting 2D points by each camera's crop."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cam_lines = ["# Camera list"]
    for cid in sorted(cameras):
        cam = cameras[cid]
        params = " ".join(repr(float(x)) for x in cam.params)
        cam_lines.append(f"{cid} {cam.model} {int(cam.width)} {int(cam.height)} {params}")
    (tmp_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    img_lines = ["# Image list with two lines of data per image"]
    for iid in sorted(images):
        im = images[iid]
        left, top = crops[im.camera_id][0], crops[im.camera_id][1]
        q = " ".join(repr(float(x)) for x in im.qvec)
        t = " ".join(repr(float(x)) for x in im.tvec)
        img_lines.append(f"{iid} {q} {t} {im.camera_id} {im.name}")
        toks = [
            f"{float(x) - left!r} {float(y) - top!r} {int(pid)}"
            for (x, y), pid in zip(im.xys, im.point3d_ids)
        ]
        img_lines.append(" ".join(toks))
    (tmp_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    pt_lines = ["# 3D point list"]
    for pt in points:
        track = " ".join(f"{iid} {pidx}" for iid, pidx in pt.track)
        pt_lines.append(
            f"{pt.point_id} {float(pt.xyz[0])!r} {float(pt.xyz[1])!r} {float(pt.xyz[2])!r} "
            f"{int(pt.rgb[0])} {int(pt.rgb[1])} {int(pt.rgb[2])} {float(pt.error)!r} {track}".rstrip()
        )
    (tmp_dir / "points3D.txt").write_text("\n".join(pt_lines) + "\n", encoding="utf-8")


def write_shared_undistorted_model(
    colmap: str,
    out_dir: Path,
    cameras: dict[int, ColmapCamera],
    images: dict[int, ColmapImage],
    points: Sequence[ColmapPoint3D],
    crops: dict[int, tuple[int, int, int, int]],
    tmp_dir: Path,
    log_dir: Path,
) -> None:
    """Write the shared centered-PINHOLE model as .bin (3DGS reads camera/pose/points3D)."""
    write_colmap_text_model(tmp_dir, cameras, images, points, crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command([colmap, "model_converter", "--input_path", str(tmp_dir), "--output_path", str(out_dir), "--output_type", "BIN"], log_dir / "shared_model_converter.log")


def undistort_and_export(colmap: str, paths: FramePaths, args, model_dir: Path, export_ctx: ExportContext) -> None:
    """Undistort + re-center-crop a frame's images into the shared 3DGS layout.

    The first call (reference frame) also computes the per-camera centering crops and
    writes the shared centered-PINHOLE sparse model; later calls reuse the crops.
    """
    undist_dir = paths.frame_out / "undist"
    run_undistort(colmap, paths.image_dir, model_dir, undist_dir, paths.log_dir, args)
    src_dir = undist_dir / "images"

    if export_ctx.crop_by_name is None:
        ucams, uimgs, upoints = read_colmap_model_any(colmap, undist_dir / "sparse", paths.frame_out / "undist_txt", paths.log_dir)
        crop_by_cam: dict[int, tuple[int, int, int, int]] = {}
        recentered: dict[int, ColmapCamera] = {}
        for cid, cam in ucams.items():
            fx, fy, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2]), float(cam.params[3])
            spec = center_crop_spec(cam.width, cam.height, cx, cy)
            crop_by_cam[cid] = spec
            _, _, w, h = spec
            recentered[cid] = ColmapCamera(cid, "PINHOLE", w, h, np.asarray([fx, fy, w / 2.0, h / 2.0], dtype=np.float64))
        export_ctx.crop_by_name = {im.name: crop_by_cam[im.camera_id] for im in uimgs.values()}
        write_shared_undistorted_model(colmap, export_ctx.sparse_dir, recentered, uimgs, upoints, crop_by_cam, paths.frame_out / "shared_txt", paths.log_dir)
        LOGGER.info("export: wrote shared undistorted model -> %s", export_ctx.sparse_dir)

    out_img_dir = export_ctx.images_root / paths.output_name
    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (src, out_img_dir / src.name, export_ctx.crop_by_name[src.name])
        for src in sorted(src_dir.iterdir())
        if src.name in export_ctx.crop_by_name
    ]
    # cv2 read/crop/write releases the GIL, so threads give real multi-core + disk overlap.
    workers = int(getattr(args, "export_workers", 0)) or auto_workers("export")
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda j: crop_image_file(j[0], j[1], j[2]), jobs))
    else:
        for src, dst, spec in jobs:
            crop_image_file(src, dst, spec)
    LOGGER.info("export: %s -> %d undistorted+centered images", paths.output_name, len(jobs))


@dataclass(frozen=True)
class RigPose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int


@dataclass(frozen=True)
class FramePaths:
    output_name: str
    frame_out: Path
    log_dir: Path
    image_dir: Path
    database_path: Path
    sparse_dir: Path
    model_txt_dir: Path
    output_ply: Path
    dense_dir: Path
    dense_ply: Path


def make_frame_paths(frame_name: str, args) -> FramePaths:
    output_name = frame_name
    frame_out = args.output_root / "_workspace" / output_name
    return FramePaths(
        output_name=output_name,
        frame_out=frame_out,
        log_dir=frame_out / "logs",
        image_dir=frame_out / "images",
        database_path=frame_out / "database.db",
        sparse_dir=frame_out / "sparse",
        model_txt_dir=frame_out / "model_txt",
        output_ply=args.output_root / "points" / f"{output_name}.ply",
        dense_dir=frame_out / "dense",
        dense_ply=args.output_root / "dense_points_cloud" / f"{output_name}.ply",
    )


def prepare_frame_workspace(paths: FramePaths, args) -> None:
    for folder in (paths.frame_out, paths.log_dir, paths.output_ply.parent):
        folder.mkdir(parents=True, exist_ok=True)
    if args.force and paths.frame_out.exists():
        for child in (paths.image_dir, paths.sparse_dir, paths.model_txt_dir, paths.dense_dir):
            if child.exists():
                shutil.rmtree(child)
        if paths.database_path.exists():
            paths.database_path.unlink()
        final_sparse_root = args.output_root / "sparse" / paths.output_name
        if final_sparse_root.exists():
            shutil.rmtree(final_sparse_root)
        if paths.output_ply.exists():
            paths.output_ply.unlink()


def make_matcher(args) -> SuperGlueMatcher:
    return SuperGlueMatcher(
        superglue_root=args.superglue_root,
        weights=args.superglue,
        resize=args.resize,
        resize_float=args.resize_float,
        max_keypoints=args.max_keypoints,
        keypoint_threshold=args.keypoint_threshold,
        nms_radius=args.nms_radius,
        sinkhorn_iterations=args.sinkhorn_iterations,
        match_threshold=args.match_threshold,
        device=args.device,
        fp16=args.fp16,
        tf32=args.tf32,
    )


def match_frame_pairs(
    image_names: Sequence[str],
    image_infos: dict[str, ImageInfo],
    matcher: SuperGlueMatcher,
    args,
    output_name: str,
    pairs: Sequence[tuple[str, str]] | None = None,
    resize: Sequence[int] | None = None,
) -> tuple[list[PairMatch], dict]:
    if pairs is None:
        manual_layout = getattr(args, "manual_camera_layout", None)
        if manual_layout is not None:
            pairs = build_manual_camera_pairs(
                image_names,
                manual_layout,
                getattr(args, "camera_name_mapping", None),
                args.camera_layout_unlisted,
            )
        else:
            layout_order = getattr(args, "layout_order", None)
        if manual_layout is None and layout_order:
            pairs = build_layout_pairs(image_names, layout_order, args.layout_window, args.layout_ring)
        elif manual_layout is None:
            pairs = build_pairs(image_names, args.pair_mode, args.pair_window, args.loop_pairs, args.pairs_file)
    blacklist = getattr(args, "pair_blacklist_set", None)
    if blacklist:
        pairs = filter_blacklisted_pairs(pairs, blacklist)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    LOGGER.info("%s: %d image pairs", output_name, len(pairs))

    # Detect SuperPoint features once per image, then reuse across all of its pairs.
    # Image decode (CPU/disk) is prefetched on worker threads so it overlaps the GPU
    # SuperPoint pass; only the GPU encode runs on this (CUDA) thread.
    used_names = sorted({n for pair in pairs for n in pair}, key=natural_key)
    feats: dict[str, dict] = {}
    workers = max(1, int(getattr(args, "decode_workers", 4)))
    mask_root = getattr(args, "mask_root", None)
    masked_out = 0

    def post_detect(name: str, feat: dict) -> dict:
        nonlocal masked_out
        if mask_root is not None:
            mask = load_camera_mask(mask_root, name)
            if mask is not None:
                before = int(feat["keypoints"][0].shape[0])
                feat = apply_feature_mask(feat, mask)
                masked_out += before - int(feat["keypoints"][0].shape[0])
        return feat

    dbar = make_pbar(len(used_names), f"detect {output_name}", "img", getattr(args, "progress", True), leave=False, position=1)
    if workers > 1 and len(used_names) > 1:
        prefetch = max(workers, 2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: dict = {}
            next_idx = 0
            for done_idx, name in enumerate(used_names):
                while next_idx < len(used_names) and len(pending) < prefetch:
                    n = used_names[next_idx]
                    pending[n] = pool.submit(matcher._decode, image_infos[n].path, resize)
                    next_idx += 1
                feats[name] = post_detect(name, matcher._encode(pending.pop(name).result()))
                dbar.update(1)
    else:
        for name in used_names:
            feats[name] = post_detect(name, matcher.detect(image_infos[name].path, resize))
            dbar.update(1)
    dbar.close()
    if masked_out:
        LOGGER.info("%s: masks removed %d keypoints across %d images", output_name, masked_out, len(used_names))

    pair_matches: list[PairMatch] = []
    raw_total = 0
    kept_total = 0
    skipped_pairs = 0
    min_inlier_ratio = float(getattr(args, "min_inlier_ratio", 0.0))
    filter_workers = max(1, int(getattr(args, "filter_workers", 2)))

    def consume(entry) -> None:
        """Collect one filtered pair (FIFO order, so pair_matches keeps the pair order)."""
        nonlocal kept_total, skipped_pairs
        name0, name1, fut = entry
        kept = fut.result() if hasattr(fut, "result") else fut
        kept_total += len(kept)
        if len(kept) >= args.min_matches:
            pair_matches.append(PairMatch(name0=name0, name1=name1, matches=kept))
        else:
            skipped_pairs += 1

    bar = make_pbar(len(pairs), f"match {output_name}", "pair", getattr(args, "progress", True), leave=False, position=1)
    # The CPU-side RANSAC filter (cv2 releases the GIL) runs on worker threads so it
    # overlaps the GPU SuperGlue matching of subsequent pairs; a bounded backlog caps RAM.
    pool = ThreadPoolExecutor(max_workers=filter_workers) if filter_workers > 1 and len(pairs) > 1 else None
    inflight: deque = deque()
    try:
        for idx, (name0, name1) in enumerate(pairs, start=1):
            if args.verbose and (idx == 1 or idx % args.log_every == 0):
                LOGGER.info("%s: matching pair %d/%d (%s, %s)", output_name, idx, len(pairs), name0, name1)
            raw = matcher.match_cached(feats[name0], feats[name1])
            raw_total += len(raw)
            if pool is not None:
                inflight.append((name0, name1, pool.submit(
                    filter_corrs, raw, image_infos[name0], image_infos[name1], args.min_matches,
                    args.ransac, args.ransac_max_error, args.ransac_confidence, min_inlier_ratio,
                )))
                while len(inflight) > filter_workers * 4:
                    consume(inflight.popleft())
            else:
                consume((name0, name1, filter_corrs(
                    raw, image_infos[name0], image_infos[name1], args.min_matches,
                    args.ransac, args.ransac_max_error, args.ransac_confidence, min_inlier_ratio,
                )))
            bar.update(1)
        while inflight:
            consume(inflight.popleft())
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    bar.close()

    match_stats = {
        "num_candidate_pairs": len(pairs),
        "num_valid_pairs": len(pair_matches),
        "num_skipped_pairs": skipped_pairs,
        "raw_matches": raw_total,
        "filtered_matches": kept_total,
    }
    return pair_matches, match_stats


def write_registration_report(
    report_root: Path,
    output_name: str,
    image_names: Sequence[str],
    candidate_pairs: Sequence[tuple[str, str]],
    pair_matches: Sequence[PairMatch],
    registered_names: set[str],
    initial_registered_names: set[str],
    rescue_pairs: Sequence[tuple[str, str]],
    rescue_matches: Sequence[PairMatch],
    colocated: dict[str, tuple[str, ...]],
) -> dict:
    """Persist an actionable per-camera registration diagnosis outside the workspace."""
    report_root.mkdir(parents=True, exist_ok=True)
    all_candidates = normalize_pairs([*candidate_pairs, *rescue_pairs], image_names)
    candidate_neighbours = {name: set() for name in image_names}
    for name0, name1 in all_candidates:
        candidate_neighbours[name0].add(name1)
        candidate_neighbours[name1].add(name0)

    valid_edges: dict[str, list[tuple[str, int]]] = {name: [] for name in image_names}
    for pair in pair_matches:
        count = int(len(pair.matches))
        valid_edges[pair.name0].append((pair.name1, count))
        valid_edges[pair.name1].append((pair.name0, count))

    image_entries = []
    for name in image_names:
        neighbours = sorted(candidate_neighbours[name], key=natural_key)
        valid = sorted(valid_edges[name], key=lambda item: natural_key(item[0]))
        registered_valid = [(other, count) for other, count in valid if other in registered_names]
        same_position = list(colocated.get(name, ()))
        status = "registered" if name in registered_names else "unregistered"
        reason_code: str | None = None
        reason: str | None = None
        if status == "unregistered":
            if not neighbours:
                reason_code = "NO_CANDIDATE_PAIRS"
                reason = "相机布局没有给该图像提供候选邻居。"
            elif not valid:
                reason_code = "NO_VALID_MATCH_PAIRS"
                reason = "候选邻居存在，但 SuperGlue + RANSAC 后没有一对达到有效匹配阈值。"
            elif not registered_valid:
                reason_code = "NO_REGISTERED_MATCHED_NEIGHBOURS"
                reason = "有二维匹配，但有效邻居均未进入最终模型，无法形成稳定的二维到三维对应。"
            else:
                reason_code = "COLMAP_POSE_REGISTRATION_FAILED"
                reason = (
                    "与已注册邻居存在有效二维匹配，但 COLMAP 仍未通过绝对位姿注册；"
                    "通常表示可复用的三维点不足、视差过小、焦距初值偏差较大，或几何对应不一致。"
                )
            if same_position:
                reason += " 同格位相机为 " + ", ".join(same_position) + "；同位置匹配本身几乎没有三角化基线。"
        image_entries.append(
            {
                "name": name,
                "status": status,
                "candidate_pair_count": len(neighbours),
                "valid_pair_count": len(valid),
                "filtered_match_count": sum(count for _other, count in valid),
                "registered_valid_neighbour_count": len(registered_valid),
                "same_position_images": same_position,
                "valid_pairs": [
                    {"other": other, "matches": count, "other_registered": other in registered_names}
                    for other, count in valid
                ],
                "reason_code": reason_code,
                "likely_reason": reason,
            }
        )

    missing = [entry for entry in image_entries if entry["status"] == "unregistered"]
    rescued = sorted(registered_names - initial_registered_names, key=natural_key)
    report = {
        "frame": output_name,
        "total_images": len(image_names),
        "registered_images": len(registered_names),
        "unregistered_images": [entry["name"] for entry in missing],
        "initial_unregistered_images": sorted(set(image_names) - initial_registered_names, key=natural_key),
        "rescued_images": rescued,
        "rescue_candidate_pairs": len(rescue_pairs),
        "rescue_valid_pairs": len(rescue_matches),
        "rescue_filtered_matches": sum(len(pair.matches) for pair in rescue_matches),
        "images": image_entries,
    }
    json_path = report_root / f"{output_name}.json"
    text_path = report_root / f"{output_name}.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"相机注册报告 / Camera registration report: {output_name}",
        f"注册结果: {len(registered_names)}/{len(image_names)}",
        f"自动补救: {len(rescue_pairs)} 个候选对, {len(rescue_matches)} 个有效对, "
        f"补回 {len(rescued)} 张 ({', '.join(rescued) if rescued else '无'})",
        "",
    ]
    if not missing:
        lines.append("全部图像均已注册。")
    else:
        lines.append("未注册图像:")
        for entry in missing:
            lines.extend(
                [
                    f"- {entry['name']}",
                    f"  原因代码: {entry['reason_code']}",
                    f"  可能原因: {entry['likely_reason']}",
                    f"  候选对/有效对/有效匹配: {entry['candidate_pair_count']}/"
                    f"{entry['valid_pair_count']}/{entry['filtered_match_count']}",
                    "  有效邻居: "
                    + (", ".join(f"{pair['other']}({pair['matches']})" for pair in entry["valid_pairs"]) or "无"),
                ]
            )
    # UTF-8 BOM keeps Chinese readable in Windows Notepad and Windows PowerShell 5.
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return report


def finalize_frame(
    model_dir: Path,
    paths: FramePaths,
    frame_dir: FrameInput,
    args,
    colmap: str,
    extra_stats: dict,
    num_images: int,
    export_ctx: ExportContext | None = None,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    export_model_txt(colmap, model_dir, paths.model_txt_dir, paths.log_dir)
    cameras, images, points = read_colmap_text_model(paths.model_txt_dir)
    raw_num_points = len(points)
    points = filter_points(points, args.ply_min_track_len, args.ply_max_reproj_error)
    LOGGER.info("%s: %d/%d points kept after track/error filtering", paths.output_name, len(points), raw_num_points)

    # In the integrated WAFT mode, velocity is computed AFTER flows are generated (Phase 3,
    # undistorted space). We still write a plain points/<N>.ply now (so progress is visible and
    # an interrupted run keeps its points) plus a cache of observing cameras; Phase 3 writes a
    # separate velocities/<N>.npz and updates the PLY only for frames with configured flow.
    # Legacy mode (external flow, no --compute_flow) computes it inline.
    defer_velocity = args.compute_flow and args.compute_velocity
    velocity_result = None
    velocity_has_flow_files = False
    if defer_velocity:
        write_point_cache(args.output_root / "_flowcache" / f"{paths.output_name}.npz", points, images)
    elif args.compute_velocity:
        flow_dir = args.flows_root / frame_dir.name
        if not flow_dir.exists():
            flow_dir = args.flows_root / paths.output_name
        if not flow_dir.exists():
            flow_dir = args.flows_root
        velocity_has_flow_files = flow_dir.exists() and any(flow_dir.glob("*.npy"))
        velocity_result = estimate_flow_velocity_for_points(points, cameras, images, flow_dir, args)
        if velocity_has_flow_files:
            xyz = np.array([p.xyz for p in points], dtype=np.float32).reshape(len(points), 3)
            rgb = np.array([p.rgb for p in points], dtype=np.uint8).reshape(len(points), 3)
            write_velocity_npz(
                args.output_root / "velocities" / f"{paths.output_name}.npz",
                xyz, rgb, velocity_result, paths.output_name, None, 1, args.velocity_dt,
            )
    write_points_ply(paths.output_ply, points, velocity_result, ascii=args.ply_ascii)

    if args.undistort:
        # In --static_rig export_ctx is shared (one camera set); otherwise make a per-frame one.
        ctx = export_ctx if export_ctx is not None else ExportContext(
            images_root=args.output_root / "undistorted" / "images",
            sparse_dir=args.output_root / "undistorted" / "sparse" / paths.output_name / "0",
        )
        undistort_and_export(colmap, paths, args, model_dir, ctx)

    if args.dense:
        paths.dense_ply.parent.mkdir(parents=True, exist_ok=True)
        run_dense(colmap, paths.image_dir, model_dir, paths.dense_dir, paths.dense_ply, paths.log_dir, args)

    stats = {
        "frame": frame_dir.name,
        "output_name": paths.output_name,
        "sparse_ply": str(paths.output_ply),
        "dense_ply": str(paths.dense_ply) if args.dense else None,
        "num_images": num_images,
        **extra_stats,
        **parse_points3d_stats(paths.model_txt_dir / "points3D.txt"),
        "num_filtered_out_points": raw_num_points - len(points),
        "num_ply_points": len(points),
        "velocity_valid_points": int(velocity_result.valid.sum()) if velocity_result is not None else 0,
    }
    if args.keep_workspace:
        (paths.frame_out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        shutil.rmtree(paths.frame_out, ignore_errors=True)
        try:
            paths.frame_out.parent.rmdir()
        except OSError:
            pass
    return stats, cameras, images, points


def reconstruct_frame(
    frame_dir: FrameInput,
    frame_name: str,
    args,
    colmap: str,
    matcher: SuperGlueMatcher,
    export_ctx: ExportContext | None = None,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    paths = make_frame_paths(frame_name, args)
    prepare_frame_workspace(paths, args)

    src_images = list_images(frame_dir, args.max_images)
    staged_images = stage_images(src_images, paths.image_dir, copy_images=args.copy_images)
    LOGGER.info("%s: %d images", paths.output_name, len(staged_images))

    reset_database(colmap, paths.database_path, paths.log_dir)
    image_infos = create_colmap_database(paths.database_path, staged_images, args.camera_model, args.focal_factor, args.single_camera)

    image_names = [p.name for p in staged_images]
    manual_layout = getattr(args, "manual_camera_layout", None)
    if manual_layout is not None:
        candidate_pairs = build_manual_camera_pairs(
            image_names,
            manual_layout,
            getattr(args, "camera_name_mapping", None),
            args.camera_layout_unlisted,
        )
    elif getattr(args, "layout_order", None):
        candidate_pairs = build_layout_pairs(
            image_names, args.layout_order, args.layout_window, args.layout_ring
        )
    else:
        candidate_pairs = build_pairs(
            image_names, args.pair_mode, args.pair_window, args.loop_pairs, args.pairs_file
        )
    if getattr(args, "pair_blacklist_set", None):
        candidate_pairs = filter_blacklisted_pairs(candidate_pairs, args.pair_blacklist_set)
    if args.max_pairs:
        candidate_pairs = candidate_pairs[: args.max_pairs]

    pair_matches, match_stats = match_frame_pairs(
        image_names, image_infos, matcher, args, paths.output_name, pairs=candidate_pairs
    )
    if not pair_matches:
        raise RuntimeError(f"{paths.output_name}: SuperGlue produced no valid pairs")

    # Optional heuristic for look-alike cameras (symmetric rigs / repetitive texture).
    # It deliberately stays opt-in because the provisional focal length and unrefined
    # intrinsics used here can make valid wide-angle pairs look cycle-inconsistent.
    if getattr(args, "cycle_filter", False) and len(pair_matches) >= 8:
        pair_matches, cycle_diag, n_dropped = cycle_filter_pairs(
            pair_matches, image_infos, args.focal_factor,
            args.cycle_max_rot_error, args.cycle_min_triangles,
        )
        match_stats = {**match_stats, "cycle_dropped_pairs": n_dropped}
        if n_dropped:
            LOGGER.warning(
                "%s: cycle-consistency filter dropped %d suspicious pair(s); "
                "review %s to confirm (paste into --pair_blacklist to make it permanent)",
                paths.output_name, n_dropped, args.output_root / "pair_report" / paths.output_name,
            )
        if not pair_matches:
            raise RuntimeError(f"{paths.output_name}: no pairs survived cycle-consistency filtering")
        if getattr(args, "pair_report", False):
            write_pair_report(
                args.output_root / "pair_report" / paths.output_name, cycle_diag, image_infos,
                args.pair_report_max, warn_err_deg=args.cycle_max_rot_error,
            )

    db_stats = export_matches_to_database(paths.database_path, image_infos, pair_matches, args.keypoint_quantization, args.two_view_config)
    LOGGER.info("%s: exported %s", paths.output_name, db_stats)

    init_ids = resolve_init_pair(getattr(args, "init_pair", None), image_infos, paths.output_name)
    model_dir = run_mapper(colmap, paths.database_path, paths.image_dir, paths.sparse_dir, paths.log_dir, args, init_ids)
    mode = "incremental_sfm"

    if args.epipolar_filter:
        export_model_txt(colmap, model_dir, paths.model_txt_dir, paths.log_dir)
        cameras, images, _ = read_colmap_text_model(paths.model_txt_dir)
        images_by_name = {img.name: img for img in images.values()}
        refined, n_in, n_out = epipolar_filter_pairs(pair_matches, cameras, images_by_name, args.epipolar_max_error, args.min_matches)
        if len(refined) >= 2 and n_out < n_in:
            LOGGER.info("%s: epipolar refine kept %d/%d matches; re-solving poses", paths.output_name, n_out, n_in)
            db_stats = export_matches_to_database(paths.database_path, image_infos, refined, args.keypoint_quantization, args.two_view_config)
            model_dir = run_mapper(colmap, paths.database_path, paths.image_dir, paths.sparse_dir, paths.log_dir, args, init_ids)
            mode = "incremental_sfm_epipolar_refined"
            pair_matches = refined
            match_stats = {**match_stats, "epipolar_matches_in": n_in, "epipolar_matches_kept": n_out}
        else:
            LOGGER.info("%s: epipolar refine skipped (kept %d/%d matches, %d pairs)", paths.output_name, n_out, n_in, len(refined))

    # Diagnose completeness after the normal solve, then add only one wider graph
    # ring around failed cameras.  The existing model stays fixed while COLMAP's
    # image_registrator gets another chance with more 2D-3D correspondences.
    export_model_txt(colmap, model_dir, paths.model_txt_dir, paths.log_dir)
    _cameras, initial_images, _points = read_colmap_text_model(paths.model_txt_dir)
    initial_registered_names = {image.name for image in initial_images.values()}
    missing_names = sorted(set(image_names) - initial_registered_names, key=natural_key)
    rescue_pairs: list[tuple[str, str]] = []
    rescue_matches: list[PairMatch] = []
    rescue_db_stats: dict[str, int] | None = None
    if missing_names:
        LOGGER.warning(
            "%s: registered %d/%d cameras; missing: %s",
            paths.output_name,
            len(initial_registered_names),
            len(image_names),
            ", ".join(missing_names),
        )
    if missing_names and args.registration_rescue:
        rescue_pairs = build_registration_rescue_pairs(
            image_names,
            candidate_pairs,
            missing_names,
            initial_registered_names,
            args.registration_rescue_hops,
            args.registration_rescue_max_pairs_per_image,
        )
        if rescue_pairs:
            LOGGER.info(
                "%s: registration rescue matching %d wider-neighbour pair(s) for %d camera(s)",
                paths.output_name,
                len(rescue_pairs),
                len(missing_names),
            )
            rescue_matches, rescue_match_stats = match_frame_pairs(
                image_names,
                image_infos,
                matcher,
                args,
                f"{paths.output_name} rescue",
                pairs=rescue_pairs,
            )
            match_stats = {
                **match_stats,
                **{f"registration_rescue_{key}": value for key, value in rescue_match_stats.items()},
            }
            if rescue_matches:
                rescue_db_stats = append_matches_to_database(
                    paths.database_path,
                    image_infos,
                    rescue_matches,
                    args.keypoint_quantization,
                    args.two_view_config,
                )
                LOGGER.info("%s: registration rescue appended %s", paths.output_name, rescue_db_stats)
                rescued_model = paths.frame_out / "rescue_registered"
                try:
                    rescue_input_model = prepare_model_for_appended_keypoints(
                        paths.model_txt_dir,
                        paths.database_path,
                        paths.frame_out / "rescue_input_model",
                    )
                    run_image_registrator(
                        colmap,
                        paths.database_path,
                        rescue_input_model,
                        rescued_model,
                        paths.log_dir,
                        args,
                    )
                    rescued_count = model_num_registered_images(rescued_model)
                    if rescued_count > len(initial_registered_names):
                        LOGGER.info(
                            "%s: image registrator improved camera count %d -> %d",
                            paths.output_name,
                            len(initial_registered_names),
                            rescued_count,
                        )
                        model_dir = rescued_model
                        mode += "_registration_rescued"
                    else:
                        LOGGER.warning(
                            "%s: image registrator did not add a camera; trying a fresh mapper solve",
                            paths.output_name,
                        )
                except (CommandError, RuntimeError, ValueError) as exc:
                    LOGGER.warning("%s: image registrator rescue failed: %s", paths.output_name, exc)

                if model_num_registered_images(model_dir) == len(initial_registered_names) and args.registration_rescue_remap:
                    remap_root = paths.frame_out / "rescue_remap"
                    remapped_model = run_mapper(
                        colmap,
                        paths.database_path,
                        paths.image_dir,
                        remap_root,
                        paths.log_dir,
                        args,
                        init_ids,
                    )
                    remapped_count = model_num_registered_images(remapped_model)
                    if remapped_count >= model_num_registered_images(model_dir):
                        model_dir = remapped_model
                        mode += "_registration_remapped"
            else:
                LOGGER.warning(
                    "%s: wider-neighbour rescue produced no valid SuperGlue/RANSAC pair",
                    paths.output_name,
                )
        else:
            LOGGER.warning(
                "%s: no additional graph neighbours are available for registration rescue",
                paths.output_name,
            )

    export_model_txt(colmap, model_dir, paths.model_txt_dir, paths.log_dir)
    _cameras, final_images, _points = read_colmap_text_model(paths.model_txt_dir)
    registered_names = {image.name for image in final_images.values()}
    combined_matches = [*pair_matches, *rescue_matches]
    report = write_registration_report(
        args.output_root / "registration_report",
        paths.output_name,
        image_names,
        candidate_pairs,
        combined_matches,
        registered_names,
        initial_registered_names,
        rescue_pairs,
        rescue_matches,
        manual_colocated_images(
            image_names, manual_layout, getattr(args, "camera_name_mapping", None)
        ),
    )
    if report["unregistered_images"]:
        LOGGER.warning(
            "%s: final registration %d/%d; see %s",
            paths.output_name,
            len(registered_names),
            len(image_names),
            args.output_root / "registration_report" / f"{paths.output_name}.txt",
        )
    else:
        LOGGER.info("%s: all %d cameras registered", paths.output_name, len(image_names))

    extra_stats = {
        **match_stats,
        "database_export": db_stats,
        "registration_rescue_database_append": rescue_db_stats,
        "num_registered_images": len(registered_names),
        "unregistered_images": report["unregistered_images"],
        "rescued_images": report["rescued_images"],
        "registration_report": str(args.output_root / "registration_report" / f"{paths.output_name}.json"),
        "mode": mode,
    }
    return finalize_frame(model_dir, paths, frame_dir, args, colmap, extra_stats, len(staged_images), export_ctx)


def create_rig_database(
    database_path: Path,
    staged_images: Sequence[Path],
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
) -> dict[str, ImageInfo]:
    """Create a database whose camera/image ids match the reference reconstruction.

    Only images that were registered in the reference rig solve are inserted, with
    their reference intrinsics and explicit ids so ``point_triangulator`` can map the
    database keypoints onto the fixed poses.
    """
    con = sqlite3.connect(database_path)
    try:
        image_infos: dict[str, ImageInfo] = {}
        inserted_cameras: set[int] = set()
        for src in staged_images:
            name = src.name
            pose = ref_pose_by_name.get(name)
            if pose is None:
                continue
            cam = ref_cameras[pose.camera_id]
            if pose.camera_id not in inserted_cameras:
                con.execute(
                    "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
                    (pose.camera_id, CAMERA_MODEL_IDS[cam.model], cam.width, cam.height, array_to_blob(np.asarray(cam.params, dtype=np.float64)), 1),
                )
                inserted_cameras.add(pose.camera_id)
            con.execute("INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)", (pose.image_id, name, pose.camera_id))
            image_infos[name] = ImageInfo(name, src, cam.width, cam.height, pose.image_id, pose.camera_id)
        con.commit()
        return image_infos
    finally:
        con.close()


def write_skeleton_model(
    skeleton_dir: Path,
    image_infos: dict[str, ImageInfo],
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
) -> None:
    """Write a COLMAP text model with known poses but no 3D points for triangulation."""
    if skeleton_dir.exists():
        shutil.rmtree(skeleton_dir)
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    used_camera_ids = sorted({info.camera_id for info in image_infos.values()})
    cam_lines = ["# Camera list", f"# Number of cameras: {len(used_camera_ids)}"]
    for cid in used_camera_ids:
        cam = ref_cameras[cid]
        params = " ".join(repr(float(x)) for x in cam.params)
        cam_lines.append(f"{cid} {cam.model} {int(cam.width)} {int(cam.height)} {params}")
    (skeleton_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    img_lines = ["# Image list with two lines of data per image"]
    for name, info in sorted(image_infos.items(), key=lambda kv: kv[1].image_id):
        pose = ref_pose_by_name[name]
        qstr = " ".join(repr(float(x)) for x in pose.qvec)
        tstr = " ".join(repr(float(x)) for x in pose.tvec)
        img_lines.append(f"{pose.image_id} {qstr} {tstr} {pose.camera_id} {name}")
        img_lines.append("")  # empty POINTS2D line (filled in by point_triangulator)
    (skeleton_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    (skeleton_dir / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")


def run_point_triangulator(
    colmap: str,
    database_path: Path,
    image_dir: Path,
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    args,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        colmap,
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--input_path",
        str(input_dir),
        "--output_path",
        str(output_dir),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.tri_min_angle",
        str(args.mapper_tri_min_angle),
        "--Mapper.min_num_matches",
        str(args.mapper_min_num_matches),
    ]
    if getattr(args, "mapper_num_threads", 0):
        base_cmd += ["--Mapper.num_threads", str(args.mapper_num_threads)]
    # Keep every camera pose exactly equal to the shared reference solve.
    cmd = base_cmd + (["--Mapper.fix_existing_images", "1"] if args.rig_fix_poses else [])
    try:
        run_command(cmd, log_dir / "point_triangulator.log")
    except CommandError:
        if not args.rig_fix_poses:
            raise
        LOGGER.warning(
            "point_triangulator failed with --Mapper.fix_existing_images (COLMAP may be < 3.7); "
            "retrying without it. Poses may drift sub-pixel between frames."
        )
        run_command(base_cmd, log_dir / "point_triangulator.log")
    if not (output_dir / "points3D.bin").exists() and not (output_dir / "points3D.txt").exists():
        raise RuntimeError(f"point_triangulator produced no model in {output_dir}")
    return output_dir


@dataclass
class RigMatchResult:
    """Output of the GPU matching stage, consumed by the CPU triangulation stage."""
    frame_dir: FrameInput
    paths: "FramePaths"
    image_infos: dict
    pair_matches: list
    match_stats: dict


def rig_match_stage(
    frame_dir: FrameInput,
    frame_name: str,
    args,
    colmap: str,
    matcher: SuperGlueMatcher,
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
    ref_images_by_name: dict[str, ColmapImage],
    fixed_pairs: Sequence[tuple[str, str]] | None = None,
) -> RigMatchResult:
    """GPU-bound part: stage images, build the rig database, match + epipolar-filter."""
    paths = make_frame_paths(frame_name, args)
    prepare_frame_workspace(paths, args)

    src_images = list_images(frame_dir, args.max_images)
    staged_images = stage_images(src_images, paths.image_dir, copy_images=args.copy_images)

    reset_database(colmap, paths.database_path, paths.log_dir)
    image_infos = create_rig_database(paths.database_path, staged_images, ref_cameras, ref_pose_by_name)
    if len(image_infos) < 2:
        raise RuntimeError(f"{paths.output_name}: fewer than 2 rig cameras available for triangulation")
    LOGGER.info("%s: %d rig cameras with known pose", paths.output_name, len(image_infos))

    image_names = [p.name for p in staged_images if p.name in image_infos]
    pairs = None
    if fixed_pairs is not None:
        pairs = [(a, b) for a, b in fixed_pairs if a in image_infos and b in image_infos]
        if not pairs:
            raise RuntimeError(f"{paths.output_name}: no rig pairs available for this frame")
    rig_resize = args.resize if args.rig_resize is None else args.rig_resize
    pair_matches, match_stats = match_frame_pairs(image_names, image_infos, matcher, args, paths.output_name, pairs=pairs, resize=rig_resize)
    if not pair_matches:
        raise RuntimeError(f"{paths.output_name}: SuperGlue produced no valid pairs")

    if args.epipolar_filter:
        pair_matches, n_in, n_out = epipolar_filter_pairs(pair_matches, ref_cameras, ref_images_by_name, args.epipolar_max_error, args.min_matches)
        LOGGER.info("%s: epipolar filter kept %d/%d matches", paths.output_name, n_out, n_in)
        if not pair_matches:
            raise RuntimeError(f"{paths.output_name}: no matches survived epipolar filtering")
        match_stats = {**match_stats, "epipolar_matches_in": n_in, "epipolar_matches_kept": n_out}

    return RigMatchResult(frame_dir, paths, image_infos, pair_matches, match_stats)


def rig_solve_stage(
    match: RigMatchResult,
    args,
    colmap: str,
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
    export_ctx: ExportContext | None = None,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    """CPU/disk-bound part: DB export, point_triangulator, filter + undistort export.

    No GPU and no shared mutable state with other frames, so it can run on a worker
    thread while the next frame is matched on the GPU."""
    paths = match.paths
    image_infos = match.image_infos
    db_stats = export_matches_to_database(paths.database_path, image_infos, match.pair_matches, args.keypoint_quantization, args.two_view_config)
    LOGGER.info("%s: exported %s", paths.output_name, db_stats)

    skeleton_dir = paths.frame_out / "rig_input"
    write_skeleton_model(skeleton_dir, image_infos, ref_cameras, ref_pose_by_name)
    model_dir = run_point_triangulator(colmap, paths.database_path, paths.image_dir, skeleton_dir, paths.sparse_dir / "0", paths.log_dir, args)
    extra_stats = {**match.match_stats, "database_export": db_stats, "mode": "rig_triangulation"}
    return finalize_frame(model_dir, paths, match.frame_dir, args, colmap, extra_stats, len(image_infos), export_ctx)


def triangulate_frame_with_rig(
    frame_dir: FrameInput,
    frame_name: str,
    args,
    colmap: str,
    matcher: SuperGlueMatcher,
    ref_cameras: dict[int, ColmapCamera],
    ref_pose_by_name: dict[str, RigPose],
    ref_images_by_name: dict[str, ColmapImage],
    fixed_pairs: Sequence[tuple[str, str]] | None = None,
    export_ctx: ExportContext | None = None,
) -> tuple[dict, dict[int, ColmapCamera], dict[int, ColmapImage], list[ColmapPoint3D]]:
    match = rig_match_stage(
        frame_dir, frame_name, args, colmap, matcher,
        ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs,
    )
    return rig_solve_stage(match, args, colmap, ref_cameras, ref_pose_by_name, export_ctx)


# ----------------------------------------------------------------------------
# Multi-GPU data-parallel workers (one process per GPU, shared work queue).
#
# Every unit of work (a frame for matching+solve, or one (source,camera) flow
# map) is independent and writes a disjoint output file, so sharding across GPUs
# leaves the output unchanged. Each worker pins itself to one card via
# CUDA_VISIBLE_DEVICES *before* torch initialises CUDA (SuperGlueMatcher/
# WaftFlowRunner import torch lazily, so setting it at the top of the worker is
# safe). Workers are module-level functions so they pickle under 'spawn'.
# ----------------------------------------------------------------------------


def _frame_worker(gpu_id, args, colmap, ref_bundle, work_q, result_q) -> None:
    """Process frames off ``work_q`` on one GPU.

    ref_bundle is None for non-static-rig (each frame is a full independent
    reconstruct_frame); otherwise it carries the shared reference state so the
    frame is triangulated against the fixed rig poses.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_logging(args.verbose, progress=False)
    try:
        matcher = make_matcher(args)
    except Exception as exc:  # GPU/model init failed on this worker
        result_q.put(("fatal", str(gpu_id), repr(exc)))
        return

    export_ctx = None
    ref_cameras = ref_pose_by_name = ref_images_by_name = fixed_pairs = None
    if ref_bundle is not None:
        ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs, ex_info = ref_bundle
        if ex_info is not None:
            images_root, sparse_dir, crop_by_name = ex_info
            export_ctx = ExportContext(images_root=images_root, sparse_dir=sparse_dir, crop_by_name=crop_by_name)

    while True:
        frame_dir = work_q.get()
        if frame_dir is None:
            break
        name = frame_dir.name
        try:
            if ref_bundle is not None:
                stats, *_ = triangulate_frame_with_rig(
                    frame_dir, name, args, colmap, matcher,
                    ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs, export_ctx,
                )
            else:
                stats, *_ = reconstruct_frame(frame_dir, name, args, colmap, matcher)
            result_q.put(("ok", name, stats))
        except Exception as exc:
            result_q.put(("err", name, repr(exc)))


def run_frames_multigpu(frames, gpus, args, colmap, ref_bundle, frame_bar) -> list[dict]:
    """Fan a list of frame dirs out over one worker process per GPU."""
    ctx = mp.get_context("spawn")
    work_q = ctx.Queue()
    result_q = ctx.Queue()
    for frame_dir in frames:
        work_q.put(frame_dir)
    for _ in gpus:
        work_q.put(None)  # one stop sentinel per worker

    procs = [
        ctx.Process(target=_frame_worker, args=(gid, args, colmap, ref_bundle, work_q, result_q), daemon=False)
        for gid in gpus
    ]
    for p in procs:
        p.start()

    stats_list: list[dict] = []
    expected = {frame_dir.name for frame_dir in frames}
    while expected:
        try:
            kind, name, payload = result_q.get(timeout=5.0)
        except queue_mod.Empty:
            if not any(p.is_alive() for p in procs):
                break  # workers gone; avoid deadlocking on unprocessed frames
            continue
        if kind == "ok":
            stats_list.append(payload)
            expected.discard(name)
            frame_bar.update(1)
        elif kind == "err":
            LOGGER.warning("frame %s failed on a GPU worker: %s", name, payload)
            expected.discard(name)
            frame_bar.update(1)
        elif kind == "fatal":
            LOGGER.error("GPU worker (device %s) failed to initialise: %s", name, payload)
    for p in procs:
        p.join(timeout=30)
    if expected:
        LOGGER.error("%d frame(s) were not processed: %s",
                     len(expected), ", ".join(sorted(expected, key=natural_key)))
    stats_list.sort(key=lambda s: natural_key(s.get("output_name", "")))
    return stats_list


def _flow_worker(gpu_id, args, work_q, result_q) -> None:
    """Compute WAFT flow maps off ``work_q`` on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_logging(args.verbose, progress=False)
    try:
        runner = WaftFlowRunner(
            args.waft_root, args.waft_cfg, args.waft_ckpt,
            "cuda", args.waft_scale, args.flow_max_size,
        )
    except Exception as exc:
        result_q.put(("fatal", str(gpu_id), repr(exc)))
        return
    while True:
        item = work_q.get()
        if item is None:
            break
        out_path, img_a, img_b = item
        try:
            flow = runner.flow(img_a, img_b)
            np.save(out_path, flow)
            result_q.put(("ok", str(out_path), None))
        except Exception as exc:
            result_q.put(("err", str(out_path), repr(exc)))


def compute_flows_for_frames_multigpu(selected_frames: Sequence[Path], args, gpus) -> int:
    """Multi-GPU version of compute_flows_for_frames: shard (source,camera) maps
    across one worker per GPU. Output .npy files are identical to the single-GPU
    path; only the device assignment differs."""
    images_root = args.output_root / "undistorted" / "images"
    flows_root = args.output_root / "flows"
    frame_pairs = build_flow_frame_pairs(selected_frames, args.flow_frame_interval)
    if not frame_pairs:
        LOGGER.warning(
            "flow: no frame pairs for --flow_frame_interval=%d over %d selected frames",
            args.flow_frame_interval, len(selected_frames),
        )
        return 0

    # Enumerate the flat work list (mirrors the single-GPU enumeration + resume check).
    units: list[tuple[Path, Path, Path]] = []
    pair_meta: list[tuple[str, str, Path, int]] = []
    for a, b in frame_pairs:
        dir_a = images_root / a
        dir_b = images_root / b
        if not dir_a.exists() or not dir_b.exists():
            LOGGER.warning("flow: missing undistorted images for %s or %s; skipping pair", a, b)
            continue
        files_a = {p.stem: p for p in dir_a.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        files_b = {p.stem: p for p in dir_b.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        common = sorted(set(files_a) & set(files_b), key=natural_key)
        out_dir = flows_root / a
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "_pair.json"
        skip_existing_for_pair = False
        if args.skip_existing_flow and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                skip_existing_for_pair = (
                    meta.get("source_frame") == a
                    and meta.get("target_frame") == b
                    and int(meta.get("flow_frame_interval", 1)) == int(args.flow_frame_interval)
                )
            except Exception as exc:
                LOGGER.warning("flow: failed to read %s for resume check: %s", meta_path, exc)
        for cam in common:
            out_path = out_dir / f"{cam}.npy"
            if not (skip_existing_for_pair and out_path.exists()):
                units.append((out_path, files_a[cam], files_b[cam]))
        pair_meta.append((a, b, out_dir, len(common)))

    grand_total = len(units)
    LOGGER.info("flow: computing %d flow maps over %d frame pairs on %d GPUs",
                grand_total, len(pair_meta), len(gpus))
    bar = make_pbar(grand_total, "flow", "map", args.progress)

    if grand_total:
        ctx = mp.get_context("spawn")
        work_q = ctx.Queue()
        result_q = ctx.Queue()
        for u in units:
            work_q.put(u)
        for _ in gpus:
            work_q.put(None)
        procs = [ctx.Process(target=_flow_worker, args=(gid, args, work_q, result_q), daemon=False) for gid in gpus]
        for p in procs:
            p.start()
        done = 0
        errors = 0
        remaining = grand_total
        while remaining > 0:
            try:
                kind, name, payload = result_q.get(timeout=5.0)
            except queue_mod.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            if kind == "ok":
                done += 1
                remaining -= 1
                bar.update(1)
            elif kind == "err":
                errors += 1
                remaining -= 1
                bar.update(1)
                LOGGER.warning("flow failed for %s: %s", name, payload)
            elif kind == "fatal":
                LOGGER.error("flow worker (device %s) failed to initialise: %s", name, payload)
        for p in procs:
            p.join(timeout=30)
    else:
        done = 0
    bar.close()

    # Write _pair.json once per source frame (parent-side, avoids worker races).
    for a, b, out_dir, ncommon in pair_meta:
        (out_dir / "_pair.json").write_text(
            json.dumps(
                {"source_frame": a, "target_frame": b, "flow_frame_interval": int(args.flow_frame_interval)},
                indent=2,
            ),
            encoding="utf-8",
        )
    LOGGER.info("flow: wrote %d flow maps under %s", done, flows_root)
    return done


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace COLMAP matcher with SuperPoint+SuperGlue matches.")
    parser.add_argument("--images_root", type=Path, default=Path("data/twopeople/images"))
    parser.add_argument("--output_root", type=Path, default=Path("output/twopeople_superglue"))
    parser.add_argument("--superglue_root", type=Path, default=Path("colmap/SuperGluePretrainedNetwork"))
    parser.add_argument("--input_layout", choices=["frames", "cameras"], default="frames",
                        help="'frames': images_root/frame/camera.png (default); 'cameras': "
                             "images_root/camera/frame.png, transposed automatically and renamed 1.png, 2.png, ...")
    parser.add_argument("--frames", type=str, default=None, help="Frame folder names or ranges, e.g. '1' or '1:3'.")
    parser.add_argument("--max_images", type=int, default=None, help="Debug limit for images per frame.")
    parser.add_argument("--max_pairs", type=int, default=None, help="Debug limit for image pairs per frame.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing frame workspace.")
    parser.add_argument("--keep_workspace", action="store_true", help="Keep temporary images, database, logs, and text model.")
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of using hard links when possible.")

    parser.add_argument("--static_rig", action=argparse.BooleanOptionalAction, default=False,
                        help="Solve camera poses once on a reference frame, then reuse them to triangulate every "
                             "frame (point_triangulator). Use when the cameras do not move between frames.")
    parser.add_argument("--rig_ref_frame", type=str, default=None,
                        help="Frame folder name used for the reference pose solve in --static_rig "
                             "(default: first selected frame).")
    parser.add_argument("--rig_fix_poses", action=argparse.BooleanOptionalAction, default=True,
                        help="In --static_rig, freeze every camera pose to the shared reference solve "
                             "(point_triangulator --Mapper.fix_existing_images). Needs COLMAP >= 3.7.")
    parser.add_argument("--rig_pair_mode", choices=["covisibility", "same"], default="covisibility",
                        help="How non-reference frames pick image pairs in --static_rig. 'covisibility' reuses a "
                             "pose-guided pair set computed once from the reference (much faster); 'same' re-uses "
                             "--pair_mode for every frame.")
    parser.add_argument("--rig_covis_top_k", type=int, default=12,
                        help="Per camera, keep this many most-covisible neighbours for the rig pair set.")
    parser.add_argument("--rig_covis_min_shared", type=int, default=20,
                        help="Minimum shared reference 3D points for a covisibility pair to count.")
    parser.add_argument("--rig_geo_neighbors", type=int, default=6,
                        help="Per camera, also add this many nearest neighbours by viewing angle (robust to motion).")
    parser.add_argument("--rig_resize", type=int, nargs="+", default=[2560],
                        help="Resize for non-reference frames in --static_rig (poses are fixed, so a lower value is "
                             "faster). Use -1 for full resolution. The reference frame always uses --resize.")

    parser.add_argument("--epipolar_filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the estimated poses to epipolar-clean matches: the reference frame re-solves "
                             "once with cleaned matches (updates intrinsics/extrinsics), and every frame's matches "
                             "are filtered before triangulation.")
    parser.add_argument("--epipolar_max_error", type=float, default=1.5,
                        help="Max epipolar (Sampson) distance in pixels for --epipolar_filter.")

    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True,
                        help="Also write undistorted images + a distortion-free model under output/undistorted/.")
    parser.add_argument("--undistort_max_image_size", type=int, default=0,
                        help="Cap the long side of undistorted images (0 = keep full resolution).")

    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--camera_model", choices=sorted(CAMERA_MODEL_IDS), default="OPENCV")
    parser.add_argument("--focal_factor", type=float, default=1.2)
    parser.add_argument("--single_camera", action="store_true")

    parser.add_argument("--pair_mode", choices=["sequential", "exhaustive", "pairs_file"], default="exhaustive")
    parser.add_argument("--pair_window", type=int, default=5)
    parser.add_argument("--loop_pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairs_file", type=Path, default=None)

    # --- Look-alike / ambiguous scene handling (auto pruning + simple user hints) ---
    parser.add_argument(
        "--camera_layout", "--camera-layout", dest="camera_layout", type=Path, default=None,
        help="Camera-grid JSON or exported neighbour-pairs TXT from camera_grid_layout_tool.py. "
             "This is the preferred manual hint for 2D/multi-camera rigs and overrides --pair_mode.",
    )
    parser.add_argument(
        "--camera_layout_unlisted", "--camera-layout-unlisted",
        dest="camera_layout_unlisted", choices=["error", "exhaustive", "ignore"], default="error",
        help="How --camera_layout handles a current image with no manual neighbour. 'error' (default) "
             "prevents silently losing cameras; 'exhaustive' safely pairs it with all images; 'ignore' "
             "keeps it in COLMAP without matches.",
    )
    parser.add_argument("--layout_file", type=Path, default=None,
                        help="Physical camera order, one name (or comma list) per line, '#' comments, "
                             "matched by stem. When given, candidate pairs are limited to cameras within "
                             "--layout_window steps of each other: look-alike cameras on opposite sides "
                             "of the rig are never matched, and the exhaustive pair count drops sharply.")
    parser.add_argument("--layout_window", type=int, default=10,
                        help="Max distance (in --layout_file order) for two cameras to be paired.")
    parser.add_argument("--layout_ring", action=argparse.BooleanOptionalAction, default=True,
                        help="Treat the layout order as a closed ring (last camera neighbours the first).")
    parser.add_argument("--pair_blacklist", type=Path, default=None,
                        help="Text file of 'nameA nameB' lines (stems, '#' comments) that must never be "
                             "matched. pair_report/suspect_pairs.txt is written in this format for reuse.")
    parser.add_argument("--mask_root", type=Path, default=None,
                        help="Directory of per-camera masks (<camera>.png; COLMAP convention: zero pixels "
                             "= ignore keypoints). Paint out repetitive/ambiguous regions once per camera; "
                             "a static rig reuses the same masks for every frame.")
    parser.add_argument("--init_pair", type=str, nargs=2, default=None, metavar=("NAME0", "NAME1"),
                        help="Seed COLMAP mapper with this initial image pair (names or stems). Pick a "
                             "well-textured, strongly overlapping pair when automatic init picks wrong.")
    parser.add_argument("--cycle_filter", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt-in heuristic: drop matched pairs whose relative rotations do not compose "
                             "to identity around view-graph triangles. Disabled by default because rotations "
                             "estimated from provisional intrinsics can reject valid wide-angle pairs.")
    parser.add_argument("--cycle_max_rot_error", type=float, default=8.0,
                        help="A pair is dropped when even its BEST view-graph triangle has a rotation "
                             "cycle error above this many degrees.")
    parser.add_argument("--cycle_min_triangles", type=int, default=2,
                        help="Minimum triangles a pair must be scored in before it can be dropped.")
    parser.add_argument("--pair_report", action=argparse.BooleanOptionalAction, default=True,
                        help="Write output_root/pair_report/<frame>/: ranked pair table, blacklist-ready "
                             "suspect_pairs.txt, and side-by-side previews of dropped/suspicious pairs "
                             "for quick human review.")
    parser.add_argument("--pair_report_max", type=int, default=40,
                        help="Max preview images per frame in the pair report.")
    parser.add_argument("--min_inlier_ratio", type=float, default=0.0,
                        help="Reject a pair whose RANSAC inlier ratio falls below this (0 disables). "
                             "Try 0.2-0.35 on rigs with strong look-alike ambiguity.")
    parser.add_argument("--filter_workers", type=int, default=2,
                        help="Threads for CPU-side RANSAC match filtering, overlapped with GPU matching "
                             "(1 disables the overlap; output is unchanged).")

    parser.add_argument("--superglue", choices=["indoor", "outdoor"], default="outdoor")
    parser.add_argument("--resize", type=int, nargs="+", default=[2560], help="SuperGlue input resize: max_dim, width height, or -1 for full resolution.")
    parser.add_argument("--resize_float", action="store_true")
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # --- Multi-GPU data parallelism (one worker process per GPU) ---
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated CUDA device indices to fan the per-frame matching and "
                             "the WAFT flow stage across, e.g. '0,1,2,3' (or 'all'). Each GPU gets its "
                             "own worker process; frames/flow-maps are independent so output is unchanged. "
                             "Unset (default) = single-GPU behaviour via --device. Ignored with --device cpu.")
    parser.add_argument("--colmap_threads", type=int, default=0,
                        help="Threads per COLMAP mapper/point_triangulator call (--Mapper.num_threads). "
                             "0 = auto: unlimited on single-GPU, or cpu_count//num_gpus with --gpus (so "
                             "concurrent COLMAP solves don't oversubscribe the CPU).")

    # --- Hardware-utilisation / speed knobs (no algorithmic change) ---
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Run SuperPoint+SuperGlue in fp16 mixed precision on CUDA (~1.5-2x). "
                             "Keypoint coordinates stay fp32; matches differ only negligibly. "
                             "Use --no-fp16 for bit-identical full-precision matching.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 matmul/conv on Ampere+ (small speedup, tiny numeric change).")
    parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=True,
                        help="In --static_rig, overlap each frame's CPU triangulation/export with the "
                             "next frame's GPU matching. Frames are independent, so output is unchanged.")
    parser.add_argument("--decode_workers", type=int, default=4,
                        help="Threads that pre-decode images so disk/CPU decode overlaps GPU detection "
                             "(1 disables prefetch).")
    parser.add_argument("--solver_workers", type=int, default=0,
                        help="In --static_rig --pipeline, how many frame solves (COLMAP triangulate + "
                             "undistort + crop) run concurrently while the GPU matches ahead. "
                             "0 = auto from CPU cores. Keep modest: these are disk-bound.")
    parser.add_argument("--export_workers", type=int, default=0,
                        help="Threads for the per-frame undistorted-image crop/write (cv2). "
                             "0 = auto from CPU cores.")

    parser.add_argument("--compute_velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flows_root", type=Path, default=Path("data/twopeople/flows"),
                        help="Legacy: external flow directory, only used with --no-compute_flow. "
                             "With --compute_flow, flows are generated to output_root/flows/.")

    # --- Integrated WAFT optical-flow stage (one command: images -> flows/points/undistorted) ---
    parser.add_argument("--compute_flow", action=argparse.BooleanOptionalAction, default=True,
                        help="Run WAFT on the undistorted output images for configured frame anchors and "
                             "write output_root/flows/<source>/<cam>.npy aligned 1:1 with undistorted/images. "
                             "When on, velocity is computed in undistorted space from these flows.")
    parser.add_argument("--flow_frame_interval", type=int, default=1,
                        help="Frame-anchor interval for --compute_flow. 1 keeps adjacent pairs. "
                             "5 records flows for the 1st, 5th, 10th, ... selected frames and divides "
                             "the estimated velocity by 5.")
    parser.add_argument("--waft_root", type=Path, default=Path("f:/project/WAFT"),
                        help="Path to the WAFT repo (imported in-process).")
    parser.add_argument("--waft_cfg", type=Path, default=Path("config/a2/twins/chairs-things.json"),
                        help="WAFT config JSON (relative to --waft_root unless absolute).")
    parser.add_argument("--waft_ckpt", type=Path, default=Path("ckpts/a2/waftv2-ckpts/twins/zero-shot.pth"),
                        help="WAFT checkpoint (relative to --waft_root unless absolute).")
    parser.add_argument("--waft_scale", type=float, default=None,
                        help="Override WAFT inference scale (2**scale input resize); default = config value.")
    parser.add_argument("--flow_max_size", type=int, default=0,
                        help="Cap the long side fed to WAFT (0 = native undistorted resolution). Lower it if "
                             "WAFT runs out of GPU memory; flow is produced at that reduced size.")
    parser.add_argument("--flow_device", default=None,
                        help="Device for WAFT (default = --device).")
    parser.add_argument("--skip_existing_flow", action=argparse.BooleanOptionalAction, default=False,
                        help="Skip flow pairs whose .npy already exists (resume a previous run).")

    parser.add_argument("--flow_format", choices=["norm", "midnorm", "pixel"], default="pixel",
                        help="Units of the flow .npy values. 'pixel' = raw pixel displacement at the "
                             "flow-map resolution (WAFT/RAFT default; rescaled by image/flow size). "
                             "'norm' = fraction of image size; 'midnorm' = [-1,1] over the image.")
    parser.add_argument("--flow_direction", choices=["forward", "backward"], default="forward",
                        help="'forward' = flow maps frame N -> N+1 (WAFT image1->image2; what the "
                             "velocity model expects). 'backward' negates the flow so N+1->N maps can "
                             "be reused without re-running the flow.")
    parser.add_argument("--flow_scale", type=float, default=1.0)
    parser.add_argument("--velocity_dt", type=float, default=1.0)
    parser.add_argument("--velocity_min_views", type=int, default=3)
    parser.add_argument("--velocity_max_reproj_error", type=float, default=3.0)
    parser.add_argument("--velocity_min_angle", type=float, default=1.0)
    parser.add_argument("--velocity_max_magnitude", type=float, default=0.0, help="0 disables speed clipping.")

    parser.add_argument("--min_matches", type=int, default=30)
    parser.add_argument("--keypoint_quantization", type=float, default=0.25)
    parser.add_argument("--two_view_config", type=int, default=3)
    parser.add_argument("--ransac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ransac_max_error", type=float, default=1.5)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--ply_min_track_len", type=int, default=3,
                        help="Drop sparse points seen by fewer than this many images (0 disables).")
    parser.add_argument("--ply_max_reproj_error", type=float, default=2.0,
                        help="Drop sparse points whose reprojection error exceeds this (0 disables).")
    parser.add_argument("--ply_ascii", action=argparse.BooleanOptionalAction, default=True,
                        help="Write point clouds as text/ASCII PLY (openable in a text editor). "
                             "Use --no-ply_ascii for compact binary PLY.")

    parser.add_argument("--ba_refine_focal_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ba_refine_principal_point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mapper_min_num_matches", type=int, default=15)
    parser.add_argument("--mapper_init_min_num_inliers", type=int, default=30)
    parser.add_argument("--mapper_init_max_error", type=float, default=4.0)
    parser.add_argument("--mapper_abs_pose_min_num_inliers", type=int, default=20)
    parser.add_argument("--mapper_tri_min_angle", type=float, default=2.0)
    parser.add_argument("--mapper_multiple_models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--registration_rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When COLMAP omits a camera, match it to one wider ring of already registered manual-layout "
             "neighbours and retry registration automatically.",
    )
    parser.add_argument("--registration_rescue_hops", type=int, default=2,
                        help="Manual view-graph distance used by registration rescue (2 = one extra ring).")
    parser.add_argument("--registration_rescue_max_pairs_per_image", type=int, default=16,
                        help="Maximum extra wider-neighbour pairs matched for each omitted camera.")
    parser.add_argument("--registration_rescue_abs_pose_max_error", type=float, default=6.0,
                        help="COLMAP reprojection threshold in pixels for rescue absolute-pose registration.")
    parser.add_argument("--registration_rescue_abs_pose_min_inlier_ratio", type=float, default=0.1,
                        help="Minimum 2D-3D inlier ratio for rescue registration; lower than COLMAP's normal "
                             "default because manual neighbours already constrain the candidate graph.")
    parser.add_argument("--registration_rescue_max_trials", type=int, default=10,
                        help="Maximum COLMAP registration trials for a missing camera during rescue.")
    parser.add_argument("--registration_rescue_remap", action=argparse.BooleanOptionalAction, default=True,
                        help="If attaching to the existing model fails, retry a fresh mapper solve with the "
                             "supplemental pairs.")

    parser.add_argument("--dense", action="store_true", help="Run image_undistorter, patch_match_stereo, stereo_fusion.")
    parser.add_argument("--gpu_index", default="0")
    parser.add_argument("--fusion_check_num_images", type=int, default=2)
    parser.add_argument("--fusion_min_num_pixels", type=int, default=3)

    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show tqdm progress bars with ETA (frames, matching pairs, flow). "
                             "Use --no-progress for plain logs (e.g. when redirecting to a file).")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.progress)

    args.images_root = args.images_root.resolve()
    args.output_root = args.output_root.resolve()
    args.superglue_root = args.superglue_root.resolve()
    args.flows_root = args.flows_root.resolve()
    if args.velocity_dt <= 0:
        raise ValueError("--velocity_dt must be positive")
    if args.flow_frame_interval <= 0:
        raise ValueError("--flow_frame_interval must be positive")
    if args.layout_window <= 0:
        raise ValueError("--layout_window must be positive")
    if args.cycle_max_rot_error <= 0:
        raise ValueError("--cycle_max_rot_error must be positive")
    if args.cycle_min_triangles <= 0:
        raise ValueError("--cycle_min_triangles must be positive")
    if not 0.0 <= args.min_inlier_ratio <= 1.0:
        raise ValueError("--min_inlier_ratio must be between 0 and 1")
    if args.pair_report_max < 0:
        raise ValueError("--pair_report_max cannot be negative")
    if args.registration_rescue_hops < 2:
        raise ValueError("--registration_rescue_hops must be at least 2")
    if args.registration_rescue_max_pairs_per_image < 0:
        raise ValueError("--registration_rescue_max_pairs_per_image cannot be negative")
    if args.registration_rescue_abs_pose_max_error <= 0:
        raise ValueError("--registration_rescue_abs_pose_max_error must be positive")
    if not 0.0 <= args.registration_rescue_abs_pose_min_inlier_ratio <= 1.0:
        raise ValueError("--registration_rescue_abs_pose_min_inlier_ratio must be between 0 and 1")
    if args.registration_rescue_max_trials <= 0:
        raise ValueError("--registration_rescue_max_trials must be positive")
    if args.pairs_file is not None:
        args.pairs_file = args.pairs_file.resolve()
    if args.camera_layout is not None and args.layout_file is not None:
        raise ValueError("--camera_layout and legacy --layout_file cannot be used together")
    if args.camera_layout is not None and args.pairs_file is not None:
        raise ValueError("--camera_layout and --pairs_file cannot be used together")

    # User hints for look-alike scenes: parse once, reuse everywhere (workers get them
    # via the pickled args namespace).
    args.manual_camera_layout = (
        load_manual_camera_layout(args.camera_layout.resolve()) if args.camera_layout else None
    )
    if args.manual_camera_layout is not None:
        manual = args.manual_camera_layout
        settings = (
            f", {manual.neighbour_mode}-neighbour radius={manual.neighbour_radius}"
            if manual.neighbour_mode is not None else ""
        )
        LOGGER.info(
            "camera layout: %s (%d cameras, %d placed, %d pairs%s)",
            manual.source_path, len(manual.camera_ids), len(manual.placed_ids), len(manual.pairs), settings,
        )
        if args.pair_mode != "exhaustive":
            LOGGER.warning("--pair_mode=%s is ignored because --camera_layout is active", args.pair_mode)
    args.layout_order = load_layout_file(args.layout_file.resolve()) if args.layout_file else None
    if args.layout_order:
        LOGGER.info(
            "layout: %d cameras, window=%d, ring=%s -> pairs limited to physically adjacent cameras",
            len(args.layout_order), args.layout_window, args.layout_ring,
        )
    args.pair_blacklist_set = load_pair_blacklist(args.pair_blacklist.resolve()) if args.pair_blacklist else set()
    if args.pair_blacklist_set:
        LOGGER.info("pair blacklist: %d forbidden pair(s)", len(args.pair_blacklist_set))
    if args.mask_root is not None:
        args.mask_root = args.mask_root.resolve()
        if not args.mask_root.exists():
            raise FileNotFoundError(f"--mask_root does not exist: {args.mask_root}")

    colmap = resolve_colmap(args.colmap)
    LOGGER.info("COLMAP: %s", colmap)
    camera_mapping: dict[str, str] = {}
    if args.input_layout == "cameras":
        frame_dirs, camera_mapping = discover_camera_major_frames(args.images_root)
        LOGGER.info(
            "camera-major input: %d cameras x %d frames (outputs named 1.png ... %d.png)",
            len(camera_mapping), len(frame_dirs), len(camera_mapping),
        )
    else:
        frame_dirs = discover_frame_dirs(args.images_root)
    args.camera_name_mapping = camera_mapping
    selected_frames = parse_frame_selection(frame_dirs, args.frames)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if camera_mapping:
        mapping_path = args.output_root / "camera_mapping.json"
        mapping_path.write_text(
            json.dumps(
                {"images_root": str(args.images_root), "camera_files": camera_mapping},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOGGER.info("camera mapping: %s", mapping_path)

    LOGGER.info("selected frames: %s", ", ".join(p.name for p in selected_frames))

    # Resolve the multi-GPU plan. --gpus only applies to CUDA; on CPU it's ignored.
    args.gpus_list = parse_gpus(args.gpus) if args.device == "cuda" else []
    multi_gpu = len(args.gpus_list) > 1
    if multi_gpu:
        # Cap COLMAP threads per worker so concurrent solves don't oversubscribe the CPU.
        args.mapper_num_threads = args.colmap_threads or max(1, (os.cpu_count() or 4) // len(args.gpus_list))
        # Pin any parent-process CUDA work (the one-time reference solve) to the first GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus_list[0])
        LOGGER.info("multi-GPU enabled: devices %s, %d COLMAP thread(s)/worker",
                    args.gpus_list, args.mapper_num_threads)
    else:
        args.mapper_num_threads = args.colmap_threads

    # The parent only needs a matcher for the single-GPU paths or the static-rig
    # reference solve; multi-GPU workers each build their own on their card.
    need_parent_matcher = (not multi_gpu) or args.static_rig
    matcher = make_matcher(args) if need_parent_matcher else None
    all_stats = []
    frame_bar = make_pbar(len(selected_frames), "frames", "frame", args.progress)

    if args.static_rig:
        by_name = {p.name: p for p in frame_dirs}
        ref_name = args.rig_ref_frame or selected_frames[0].name
        if ref_name not in by_name:
            raise ValueError(f"--rig_ref_frame '{ref_name}' not found among frame folders")
        ref_frame = by_name[ref_name]
        export_ctx = ExportContext(
            images_root=args.output_root / "undistorted" / "images",
            sparse_dir=args.output_root / "undistorted" / "sparse" / "0",
        ) if args.undistort else None
        LOGGER.info("static rig: solving reference poses from frame '%s'", ref_name)
        frame_bar.set_postfix_str(f"ref {ref_name}")
        ref_stats, ref_cameras, ref_images, ref_points = reconstruct_frame(ref_frame, ref_name, args, colmap, matcher, export_ctx)
        all_stats.append(ref_stats)
        frame_bar.update(1)
        ref_pose_by_name = {
            img.name: RigPose(img.image_id, img.qvec, img.tvec, img.camera_id)
            for img in ref_images.values()
        }
        ref_images_by_name = {img.name: img for img in ref_images.values()}
        LOGGER.info("static rig: reference registered %d/%d cameras", len(ref_pose_by_name), ref_stats["num_images"])

        fixed_pairs = None
        ref_names = sorted(ref_pose_by_name, key=natural_key)
        if args.manual_camera_layout is not None:
            fixed_pairs = build_manual_camera_pairs(
                ref_names,
                args.manual_camera_layout,
                args.camera_name_mapping,
                args.camera_layout_unlisted,
            )
            full = len(ref_names) * (len(ref_names) - 1) // 2
            LOGGER.info(
                "static rig: reusing %d manual layout pairs for every frame "
                "(exhaustive would be %d)",
                len(fixed_pairs), full,
            )
        elif args.rig_pair_mode == "covisibility":
            fixed_pairs = build_rig_pairs(
                ref_images, ref_points, ref_names,
                args.rig_covis_top_k, args.rig_covis_min_shared, args.rig_geo_neighbors,
            )
            full = len(ref_names) * (len(ref_names) - 1) // 2
            LOGGER.info("static rig: pose-guided pair set = %d pairs (exhaustive would be %d)", len(fixed_pairs), full)
            if not fixed_pairs:
                LOGGER.warning("static rig: covisibility pair set empty; falling back to per-frame --pair_mode")
                fixed_pairs = None

        rig_frames = [f for f in selected_frames if f.name != ref_name]
        if multi_gpu and rig_frames:
            # Free the parent's reference matcher, then fan the independent per-frame
            # match+solve out over one worker process per GPU.
            matcher = None
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            LOGGER.info("multi-GPU: matching + triangulating %d frames across GPUs %s",
                        len(rig_frames), args.gpus_list)
            ex_info = (
                (export_ctx.images_root, export_ctx.sparse_dir, export_ctx.crop_by_name)
                if export_ctx is not None else None
            )
            ref_bundle = (ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs, ex_info)
            all_stats.extend(
                run_frames_multigpu(rig_frames, args.gpus_list, args, colmap, ref_bundle, frame_bar)
            )
        elif args.pipeline and len(rig_frames) > 1:
            # Overlap each frame's CPU/disk solve (triangulate + undistort + crop) with the
            # GPU matching of later frames. Frames are independent (shared fixed poses), so the
            # output is unchanged. A pool of `solver_workers` runs several disk-bound solves
            # concurrently while the GPU matches ahead; a bounded backlog caps RAM (each holds
            # that frame's pair_matches) and the GPU blocks only when the backlog is full.
            solver_workers = int(args.solver_workers) or auto_workers("solver")
            max_inflight = solver_workers + 1
            LOGGER.info("pipeline: %d solver worker(s), %d export thread(s)",
                        solver_workers, int(args.export_workers) or auto_workers("export"))
            with ThreadPoolExecutor(max_workers=solver_workers) as solver:
                inflight: deque = deque()
                for frame_dir in rig_frames:
                    frame_bar.set_postfix_str(frame_dir.name)
                    match = rig_match_stage(
                        frame_dir, frame_dir.name, args, colmap, matcher,
                        ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs,
                    )
                    while len(inflight) >= max_inflight:
                        all_stats.append(inflight.popleft().result()[0])
                        frame_bar.update(1)
                    inflight.append(solver.submit(
                        rig_solve_stage, match, args, colmap, ref_cameras, ref_pose_by_name, export_ctx,
                    ))
                while inflight:
                    all_stats.append(inflight.popleft().result()[0])
                    frame_bar.update(1)
        else:
            for frame_dir in rig_frames:
                frame_bar.set_postfix_str(frame_dir.name)
                stats, _, _, _ = triangulate_frame_with_rig(
                    frame_dir, frame_dir.name, args, colmap, matcher, ref_cameras, ref_pose_by_name, ref_images_by_name, fixed_pairs, export_ctx
                )
                all_stats.append(stats)
                frame_bar.update(1)
    elif multi_gpu:
        # Non-rig: every frame is a fully independent reconstruction, so fan them
        # out over one worker per GPU directly (no shared reference state).
        LOGGER.info("multi-GPU: reconstructing %d frames across GPUs %s",
                    len(selected_frames), args.gpus_list)
        all_stats.extend(
            run_frames_multigpu(selected_frames, args.gpus_list, args, colmap, None, frame_bar)
        )
    else:
        for frame_dir in selected_frames:
            frame_bar.set_postfix_str(frame_dir.name)
            stats, _, _, _ = reconstruct_frame(frame_dir, frame_dir.name, args, colmap, matcher)
            all_stats.append(stats)
            frame_bar.update(1)

    frame_bar.close()

    # Phase 2 (WAFT flows on undistorted images) + Phase 3 (undistorted-space velocity).
    if args.compute_flow:
        if not args.undistort:
            LOGGER.warning("--compute_flow needs --undistort (flows are computed on undistorted images); skipping")
        else:
            matcher = None  # free SuperGlue VRAM before loading WAFT (may already be None)
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            flow_ran = False
            if multi_gpu:
                # Each worker builds its own WaftFlowRunner on its card; if all fail
                # they log it and no flow files are written (velocity then no-ops).
                compute_flows_for_frames_multigpu(selected_frames, args, args.gpus_list)
                flow_ran = True
            else:
                runner = None
                try:
                    runner = WaftFlowRunner(
                        args.waft_root, args.waft_cfg, args.waft_ckpt,
                        args.flow_device or args.device, args.waft_scale, args.flow_max_size,
                    )
                except Exception as exc:
                    LOGGER.warning("WAFT flow stage skipped: %s", exc)
                if runner is not None:
                    compute_flows_for_frames(selected_frames, args, runner)
                    del runner
                    flow_ran = True
            if flow_ran:
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                if args.compute_velocity:
                    if args.static_rig:
                        run_undistorted_velocity_pass(args, colmap, selected_frames)
                    else:
                        LOGGER.warning("velocity in undistorted space currently supports --static_rig only; "
                                       "flows were written but points/*.ply have no velocity")

    if args.undistort:
        LOGGER.info("3DGS export: images=%s  shared sparse=%s",
                    args.output_root / "undistorted" / "images",
                    args.output_root / "undistorted" / "sparse" / "0")
        if args.compute_flow:
            LOGGER.info("flow export: %s (frame interval=%d)", args.output_root / "flows", args.flow_frame_interval)
        if args.compute_flow and args.compute_velocity:
            LOGGER.info("velocity export: %s", args.output_root / "velocities")
    combined = args.compute_flow and args.compute_velocity
    velocity_frame_names = {a for a, _ in build_flow_frame_pairs(selected_frames, args.flow_frame_interval)} if combined else set()
    for stats in all_stats:
        if combined:
            velocity_path = args.output_root / "velocities" / f"{stats['output_name']}.npz"
            velocity_note = str(velocity_path) if stats["output_name"] in velocity_frame_names else "skipped (no configured flow)"
            LOGGER.info("%s: ply=%s velocity=%s", stats["output_name"], stats["sparse_ply"], velocity_note)
        else:
            LOGGER.info(
                "%s: ply=%s velocity=%d/%d",
                stats["output_name"],
                stats["sparse_ply"],
                stats["velocity_valid_points"],
                stats["num_sparse_points"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
