"""Interactive camera-matrix layout editor.

The tool lets a user arrange one representative image per camera on an
irregular 2D grid, then exports the layout and physical-neighbour graph.  The
layout is camera-level metadata and can be reused for every captured frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal, Sequence

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:  # pragma: no cover - handled with a clear error in main()
    tk = None
    filedialog = messagebox = simpledialog = ttk = None

try:
    from PIL import Image, ImageOps, ImageTk
except ImportError:  # Thumbnails are optional; the layout editor still works.
    Image = ImageOps = ImageTk = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LAYOUT_FORMAT = "camera-grid-layout"
LAYOUT_VERSION = 2


def natural_key(value: str | Path) -> tuple:
    text = str(value).replace("\\", "/").lower()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


class LayoutError(ValueError):
    """Raised when a layout mutation would produce an invalid state."""


@dataclass(frozen=True)
class CameraItem:
    camera_id: str
    preview_path: str
    source_path: str
    source_kind: Literal["file", "camera_folder"] = "file"

    @property
    def image_name(self) -> str:
        return Path(self.preview_path).name

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "preview_path": self.preview_path,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "image_name": self.image_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraItem":
        camera_id = str(data.get("camera_id", "")).strip()
        if not camera_id:
            raise LayoutError("camera item is missing camera_id")
        source_kind = str(data.get("source_kind", "file"))
        if source_kind not in {"file", "camera_folder"}:
            raise LayoutError(f"unsupported camera source_kind: {source_kind}")
        return cls(
            camera_id=camera_id,
            preview_path=str(data.get("preview_path", "")),
            source_path=str(data.get("source_path", data.get("preview_path", ""))),
            source_kind=source_kind,
        )


@dataclass
class GridSlot:
    enabled: bool = True
    camera_ids: list[str] = field(default_factory=list)

    @property
    def camera_id(self) -> str | None:
        """Primary camera kept as a convenience for old callers and previews."""
        return self.camera_ids[0] if self.camera_ids else None

    @camera_id.setter
    def camera_id(self, value: str | None) -> None:
        self.camera_ids = [value] if value else []

    def to_dict(self) -> dict:
        return {"enabled": bool(self.enabled), "camera_ids": list(self.camera_ids)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "GridSlot":
        if data is None:
            return cls(enabled=False)
        if not isinstance(data, dict):
            raise LayoutError("grid slot must be an object or null")
        enabled_value = data.get("enabled", True)
        if not isinstance(enabled_value, bool):
            raise LayoutError("grid slot enabled must be true or false")
        if "camera_ids" in data:
            values = data["camera_ids"]
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise LayoutError("grid slot camera_ids must be a list of non-empty strings")
            if len(values) != len(set(values)):
                raise LayoutError("a grid slot cannot contain the same camera more than once")
            camera_ids = list(values)
        else:
            # Version 1 stored a single camera_id.  Reading it here makes old
            # customer layouts migrate transparently on their next save.
            legacy_camera = data.get("camera_id")
            camera_ids = [str(legacy_camera)] if legacy_camera else []
        return cls(enabled=enabled_value, camera_ids=camera_ids)


@dataclass(frozen=True)
class LayoutValidation:
    loaded: int
    placed: int
    active_empty_slots: int
    disabled_slots: int
    occupied_positions: int
    multi_camera_positions: int
    pair_count: int
    components: tuple[tuple[str, ...], ...]
    isolated: tuple[str, ...]
    missing_previews: tuple[str, ...]

    @property
    def connected(self) -> bool:
        return self.placed <= 1 or len(self.components) == 1

    @property
    def unplaced(self) -> int:
        return self.loaded - self.placed


class CameraGridLayout:
    """Pure data model used by both the GUI and tests."""

    def __init__(self, rows: int = 4, columns: int = 4) -> None:
        if rows <= 0 or columns <= 0:
            raise LayoutError("rows and columns must be positive")
        self.default_columns = int(columns)
        self.rows: list[list[GridSlot]] = [
            [GridSlot() for _ in range(columns)] for _ in range(rows)
        ]
        self.cameras: dict[str, CameraItem] = {}

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def reset_grid(self, rows: int, columns: int) -> None:
        if rows <= 0 or columns <= 0:
            raise LayoutError("rows and columns must be positive")
        self.default_columns = int(columns)
        self.rows = [[GridSlot() for _ in range(columns)] for _ in range(rows)]

    def set_row_length(self, row: int, length: int) -> None:
        self._check_row(row)
        if length <= 0:
            raise LayoutError("each row needs at least one slot")
        slots = self.rows[row]
        if length > len(slots):
            slots.extend(GridSlot() for _ in range(length - len(slots)))
        elif length < len(slots):
            del slots[length:]

    def insert_slot(self, row: int, column: int) -> None:
        self._check_row(row)
        if not 0 <= column <= len(self.rows[row]):
            raise LayoutError(f"column out of range: {column}")
        self.rows[row].insert(column, GridSlot())

    def remove_slot(self, row: int, column: int) -> tuple[str, ...]:
        self._check_slot(row, column)
        if len(self.rows[row]) <= 1:
            raise LayoutError("each row needs at least one slot")
        return tuple(self.rows[row].pop(column).camera_ids)

    def toggle_slot_enabled(self, row: int, column: int) -> tuple[str, ...]:
        self._check_slot(row, column)
        slot = self.rows[row][column]
        removed = tuple(slot.camera_ids)
        slot.enabled = not slot.enabled
        if not slot.enabled:
            slot.camera_ids.clear()
        return removed

    def set_slot_enabled(self, row: int, column: int, enabled: bool) -> tuple[str, ...]:
        self._check_slot(row, column)
        slot = self.rows[row][column]
        removed = tuple(slot.camera_ids) if not enabled else ()
        slot.enabled = bool(enabled)
        if not slot.enabled:
            slot.camera_ids.clear()
        return removed

    def add_camera(self, item: CameraItem) -> bool:
        existing = self.cameras.get(item.camera_id)
        if existing is not None:
            if Path(existing.source_path) == Path(item.source_path):
                return False
            raise LayoutError(f"duplicate camera id: {item.camera_id}")
        self.cameras[item.camera_id] = item
        return True

    def add_cameras(self, items: Iterable[CameraItem]) -> tuple[int, list[str]]:
        added = 0
        collisions: list[str] = []
        for item in items:
            try:
                added += int(self.add_camera(item))
            except LayoutError:
                collisions.append(item.camera_id)
        return added, collisions

    def remove_camera(self, camera_id: str) -> None:
        if camera_id not in self.cameras:
            raise LayoutError(f"unknown camera: {camera_id}")
        position = self.position_of(camera_id)
        if position is not None:
            self.rows[position[0]][position[1]].camera_ids.remove(camera_id)
        del self.cameras[camera_id]

    def rename_camera(self, old_id: str, new_id: str) -> None:
        new_id = new_id.strip()
        if not new_id or "\n" in new_id or "\r" in new_id:
            raise LayoutError("camera id cannot be empty or contain a newline")
        if old_id not in self.cameras:
            raise LayoutError(f"unknown camera: {old_id}")
        if new_id != old_id and new_id in self.cameras:
            raise LayoutError(f"duplicate camera id: {new_id}")
        item = self.cameras.pop(old_id)
        self.cameras[new_id] = CameraItem(
            new_id, item.preview_path, item.source_path, item.source_kind
        )
        position = self.position_of(old_id)
        if position is not None:
            camera_ids = self.rows[position[0]][position[1]].camera_ids
            camera_ids[camera_ids.index(old_id)] = new_id

    def position_of(self, camera_id: str) -> tuple[int, int] | None:
        for row_index, row in enumerate(self.rows):
            for column_index, slot in enumerate(row):
                if camera_id in slot.camera_ids:
                    return row_index, column_index
        return None

    def placed_camera_ids(self) -> set[str]:
        return {
            camera_id
            for row in self.rows
            for slot in row
            if slot.enabled
            for camera_id in slot.camera_ids
        }

    def place_camera(self, camera_id: str, row: int, column: int) -> None:
        if camera_id not in self.cameras:
            raise LayoutError(f"unknown camera: {camera_id}")
        self._check_slot(row, column)
        target = self.rows[row][column]
        if not target.enabled:
            raise LayoutError("cannot place a camera in a disabled gap")

        source_position = self.position_of(camera_id)
        if source_position == (row, column):
            return

        if source_position is not None:
            source = self.rows[source_position[0]][source_position[1]]
            source.camera_ids.remove(camera_id)
        target.camera_ids.append(camera_id)

    def clear_slot(self, row: int, column: int) -> tuple[str, ...]:
        self._check_slot(row, column)
        removed = tuple(self.rows[row][column].camera_ids)
        self.rows[row][column].camera_ids.clear()
        return removed

    def remove_camera_from_slot(self, camera_id: str) -> bool:
        position = self.position_of(camera_id)
        if position is None:
            return False
        self.rows[position[0]][position[1]].camera_ids.remove(camera_id)
        return True

    def active_positions(self, empty_only: bool = False, serpentine: bool = False) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for row_index, row in enumerate(self.rows):
            columns: Sequence[int] = range(len(row))
            if serpentine and row_index % 2:
                columns = reversed(range(len(row)))
            for column_index in columns:
                slot = row[column_index]
                if slot.enabled and (not empty_only or not slot.camera_ids):
                    positions.append((row_index, column_index))
        return positions

    def auto_fill(self, order: Literal["row_major", "serpentine"] = "row_major") -> list[str]:
        if order not in {"row_major", "serpentine"}:
            raise LayoutError(f"unsupported fill order: {order}")
        placed = self.placed_camera_ids()
        unplaced = sorted((camera_id for camera_id in self.cameras if camera_id not in placed), key=natural_key)
        positions = self.active_positions(empty_only=True, serpentine=order == "serpentine")
        for camera_id, (row, column) in zip(unplaced, positions):
            self.rows[row][column].camera_ids.append(camera_id)
        return unplaced[len(positions):]

    def neighbour_pairs(self, mode: Literal[4, 8] = 8, radius: int = 1) -> list[tuple[str, str]]:
        if mode not in {4, 8}:
            raise LayoutError("neighbour mode must be 4 or 8")
        if radius <= 0:
            raise LayoutError("neighbour radius must be positive")
        occupied = [
            (row_index, column_index, tuple(slot.camera_ids))
            for row_index, row in enumerate(self.rows)
            for column_index, slot in enumerate(row)
            if slot.enabled and slot.camera_ids
        ]
        pairs: list[tuple[str, str]] = []
        for _row, _column, camera_ids in occupied:
            for camera0, camera1 in combinations(camera_ids, 2):
                pairs.append(tuple(sorted((camera0, camera1), key=natural_key)))
        for (row0, col0, cameras0), (row1, col1, cameras1) in combinations(occupied, 2):
            dr, dc = abs(row0 - row1), abs(col0 - col1)
            adjacent = (
                ((dr == 0 and 0 < dc <= radius) or (dc == 0 and 0 < dr <= radius))
                if mode == 4
                else max(dr, dc) <= radius
            )
            if adjacent and (dr or dc):
                for camera0 in cameras0:
                    for camera1 in cameras1:
                        pairs.append(tuple(sorted((camera0, camera1), key=natural_key)))
        return sorted(set(pairs), key=lambda pair: (natural_key(pair[0]), natural_key(pair[1])))

    def validate(self, mode: Literal[4, 8] = 8, radius: int = 1) -> LayoutValidation:
        placed = self.placed_camera_ids()
        pairs = self.neighbour_pairs(mode, radius)
        adjacency: dict[str, set[str]] = {camera_id: set() for camera_id in placed}
        for camera0, camera1 in pairs:
            adjacency[camera0].add(camera1)
            adjacency[camera1].add(camera0)

        components: list[tuple[str, ...]] = []
        unseen = set(placed)
        while unseen:
            seed = min(unseen, key=natural_key)
            stack = [seed]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)
            unseen -= component
            components.append(tuple(sorted(component, key=natural_key)))

        active_empty = sum(slot.enabled and not slot.camera_ids for row in self.rows for slot in row)
        disabled = sum(not slot.enabled for row in self.rows for slot in row)
        occupied_positions = sum(slot.enabled and bool(slot.camera_ids) for row in self.rows for slot in row)
        multi_camera_positions = sum(
            slot.enabled and len(slot.camera_ids) > 1 for row in self.rows for slot in row
        )
        isolated = tuple(sorted((camera for camera, neighbours in adjacency.items() if not neighbours), key=natural_key))
        missing_previews = tuple(
            sorted(
                (camera_id for camera_id, item in self.cameras.items() if not Path(item.preview_path).is_file()),
                key=natural_key,
            )
        )
        components.sort(key=lambda component: natural_key(component[0]) if component else ())
        return LayoutValidation(
            loaded=len(self.cameras),
            placed=len(placed),
            active_empty_slots=int(active_empty),
            disabled_slots=int(disabled),
            occupied_positions=int(occupied_positions),
            multi_camera_positions=int(multi_camera_positions),
            pair_count=len(pairs),
            components=tuple(components),
            isolated=isolated,
            missing_previews=missing_previews,
        )

    def to_dict(self) -> dict:
        return {
            "format": LAYOUT_FORMAT,
            "version": LAYOUT_VERSION,
            "default_columns": self.default_columns,
            "cameras": [self.cameras[camera_id].to_dict() for camera_id in sorted(self.cameras, key=natural_key)],
            "rows": [[slot.to_dict() for slot in row] for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraGridLayout":
        if not isinstance(data, dict):
            raise LayoutError("layout root must be a JSON object")
        if data.get("format") != LAYOUT_FORMAT:
            raise LayoutError("not a camera-grid-layout file")
        try:
            version = int(data.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise LayoutError("layout version must be an integer") from exc
        if version not in {1, LAYOUT_VERSION}:
            raise LayoutError(f"unsupported layout version: {version}")
        row_data = data.get("rows")
        if not isinstance(row_data, list) or not row_data or any(not isinstance(row, list) or not row for row in row_data):
            raise LayoutError("layout must contain at least one non-empty row")

        layout = cls(1, 1)
        try:
            layout.default_columns = int(data.get("default_columns", len(row_data[0])))
        except (TypeError, ValueError) as exc:
            raise LayoutError("default_columns must be a positive integer") from exc
        if layout.default_columns <= 0:
            raise LayoutError("default_columns must be a positive integer")
        layout.rows = [[GridSlot.from_dict(slot) for slot in row] for row in row_data]
        layout.cameras = {}
        camera_data = data.get("cameras", [])
        if not isinstance(camera_data, list):
            raise LayoutError("cameras must be a list")
        for item_data in camera_data:
            if not isinstance(item_data, dict):
                raise LayoutError("each camera item must be an object")
            layout.add_camera(CameraItem.from_dict(item_data))

        seen: set[str] = set()
        for row in layout.rows:
            for slot in row:
                if not slot.enabled and slot.camera_ids:
                    raise LayoutError("disabled gap cannot contain cameras")
                for camera_id in slot.camera_ids:
                    if camera_id not in layout.cameras:
                        raise LayoutError(f"slot refers to unknown camera: {camera_id}")
                    if camera_id in seen:
                        raise LayoutError(f"camera appears more than once: {camera_id}")
                    seen.add(camera_id)
        return layout

    @classmethod
    def load(cls, path: Path) -> "CameraGridLayout":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LayoutError(f"failed to read layout {path}: {exc}") from exc
        return cls.from_dict(data)

    def _check_row(self, row: int) -> None:
        if not 0 <= row < len(self.rows):
            raise LayoutError(f"row out of range: {row}")

    def _check_slot(self, row: int, column: int) -> None:
        self._check_row(row)
        if not 0 <= column < len(self.rows[row]):
            raise LayoutError(f"column out of range: {column}")


def camera_items_from_files(paths: Iterable[Path]) -> list[CameraItem]:
    files = sorted(
        {path.resolve() for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES},
        key=natural_key,
    )
    return [CameraItem(path.name, str(path), str(path), "file") for path in files]


def discover_camera_items(root: Path) -> list[CameraItem]:
    """Load either one preview per camera subfolder or all images in a frame folder."""
    root = root.resolve()
    if not root.is_dir():
        raise LayoutError(f"folder does not exist: {root}")
    camera_items: list[CameraItem] = []
    for folder in sorted((path for path in root.iterdir() if path.is_dir()), key=natural_key):
        images = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=natural_key,
        )
        if images:
            camera_items.append(CameraItem(folder.name, str(images[0].resolve()), str(folder.resolve()), "camera_folder"))
    if camera_items:
        return camera_items
    return camera_items_from_files(root.iterdir())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def export_layout(
    layout: CameraGridLayout,
    json_path: Path,
    neighbour_mode: Literal[4, 8] = 8,
    neighbour_radius: int = 1,
) -> list[Path]:
    """Write canonical JSON plus human-readable matrix, CSV and neighbour pairs."""
    json_path = json_path.with_suffix(".json")
    data = layout.to_dict()
    data["neighbour_settings"] = {"mode": neighbour_mode, "radius": neighbour_radius}
    data["neighbour_pairs"] = [list(pair) for pair in layout.neighbour_pairs(neighbour_mode, neighbour_radius)]
    validation = layout.validate(neighbour_mode, neighbour_radius)
    data["validation"] = {
        "connected": validation.connected,
        "components": [list(component) for component in validation.components],
        "loaded_cameras": validation.loaded,
        "placed_cameras": validation.placed,
        "occupied_positions": validation.occupied_positions,
        "multi_camera_positions": validation.multi_camera_positions,
    }
    _atomic_write_text(json_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    stem = json_path.with_suffix("")
    matrix_path = stem.with_name(stem.name + "_matrix.txt")
    matrix_lines = [
        "# Camera grid: tab-separated; cameras sharing one position are joined by ' | '.",
        "# '_' = active empty slot; '-' = disabled physical gap.",
        f"# rows={layout.row_count}",
    ]
    for row in layout.rows:
        values = ["-" if not slot.enabled else (" | ".join(slot.camera_ids) or "_") for slot in row]
        matrix_lines.append("\t".join(values))
    _atomic_write_text(matrix_path, "\n".join(matrix_lines) + "\n")

    csv_path = stem.with_name(stem.name + "_matrix.csv")
    csv_rows: list[list[str | int]] = [[
        "row",
        "column",
        "enabled",
        "slot_order",
        "slot_camera_count",
        "camera_id",
        "preview_path",
        "source_path",
    ]]
    for row_index, row in enumerate(layout.rows, 1):
        for column_index, slot in enumerate(row, 1):
            if not slot.camera_ids:
                csv_rows.append([row_index, column_index, int(slot.enabled), "", 0, "", "", ""])
                continue
            for slot_order, camera_id in enumerate(slot.camera_ids, 1):
                item = layout.cameras[camera_id]
                csv_rows.append(
                    [
                        row_index,
                        column_index,
                        int(slot.enabled),
                        slot_order,
                        len(slot.camera_ids),
                        camera_id,
                        item.preview_path,
                        item.source_path,
                    ]
                )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", delete=False, dir=csv_path.parent,
        prefix=f".{csv_path.name}.", suffix=".tmp",
    ) as stream:
        temporary_csv = Path(stream.name)
        csv.writer(stream).writerows(csv_rows)
    try:
        os.replace(temporary_csv, csv_path)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        raise

    pairs_path = stem.with_name(stem.name + "_neighbour_pairs.txt")
    pair_lines = [
        "# Camera IDs, tab-separated.",
        f"# mode={neighbour_mode} radius={neighbour_radius}",
    ]
    pair_lines.extend(f"{camera0}\t{camera1}" for camera0, camera1 in layout.neighbour_pairs(neighbour_mode, neighbour_radius))
    _atomic_write_text(pairs_path, "\n".join(pair_lines) + "\n")
    return [json_path, matrix_path, csv_path, pairs_path]


class ThumbnailCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], object] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(self, path: str, width: int, height: int):
        if Image is None or ImageTk is None:
            return None
        key = (path, int(width), int(height))
        if key in self._cache:
            return self._cache[key]
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB") if ImageOps is not None else opened.convert("RGB")
                image.thumbnail((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (max(1, width), max(1, height)), "#20242b")
                offset = ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2)
                canvas.paste(image, offset)
                photo = ImageTk.PhotoImage(canvas)
        except Exception:
            return None
        self._cache[key] = photo
        return photo


class CameraGridApp:
    HISTORY_LIMIT = 60
    COLOR_BG = "#eef3f8"
    COLOR_PANEL = "#ffffff"
    COLOR_HEADER = "#14243a"
    COLOR_ACCENT = "#2878d0"
    COLOR_ACCENT_DARK = "#1d5fa9"
    COLOR_TEXT = "#172033"
    COLOR_MUTED = "#6b7688"
    COLOR_SUCCESS = "#24a36a"
    COLOR_MULTI = "#805ad5"

    def __init__(self, root, rows: int = 4, columns: int = 4, layout_path: Path | None = None) -> None:
        self.root = root
        self.model = CameraGridLayout(rows, columns)
        self.project_path: Path | None = None
        self.dirty = False
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.thumbnail_cache = ThumbnailCache()
        self.selected_camera_id: str | None = None
        self.current_slot: tuple[int, int] | None = None
        self.drag_payload: tuple | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_window = None
        self.drag_photo = None
        self.drag_target: tuple[int, int] | None = None
        self.slot_widgets: dict[tuple[int, int], object] = {}
        self.slot_base_borders: dict[tuple[int, int], str] = {}
        self._grid_refresh_job = None
        self._suppress_setting_change = False

        self.rows_var = tk.IntVar(value=rows)
        self.columns_var = tk.IntVar(value=columns)
        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="全部")
        self.fill_order_var = tk.StringVar(value="自然逐行")
        self.neighbour_mode_var = tk.StringVar(value="8 邻域")
        self.neighbour_radius_var = tk.IntVar(value=1)
        self.cell_size_var = tk.IntVar(value=154)
        self.status_var = tk.StringVar(value="准备就绪")
        self.summary_var = tk.StringVar()

        self.root.title("相机矩阵排列工具")
        self.root.geometry("1600x950")
        self.root.minsize(1120, 700)
        self.root.configure(background=self.COLOR_BG)
        self._configure_style()
        self._build_ui()
        self._bind_shortcuts()
        self.neighbour_mode_var.trace_add("write", self._on_neighbour_settings_changed)
        self.neighbour_radius_var.trace_add("write", self._on_neighbour_settings_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if layout_path is not None:
            self._load_layout_path(layout_path, ask_discard=False)
        else:
            self.refresh_all()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        available = style.theme_names()
        for preferred in ("clam", "vista"):
            if preferred in available:
                try:
                    style.theme_use(preferred)
                except tk.TclError:
                    pass
                break
        style.configure("App.TFrame", background=self.COLOR_BG)
        style.configure("Panel.TFrame", background=self.COLOR_PANEL)
        style.configure("Panel.TLabelframe", background=self.COLOR_PANEL, bordercolor="#dbe3ed")
        style.configure("Panel.TLabelframe.Label", background=self.COLOR_PANEL, foreground=self.COLOR_TEXT)
        style.configure("Toolbar.TButton", padding=(10, 6), font=("Microsoft YaHei UI", 9))
        style.configure(
            "Primary.TButton",
            padding=(12, 7),
            background=self.COLOR_ACCENT,
            foreground="#ffffff",
            bordercolor=self.COLOR_ACCENT,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.COLOR_ACCENT_DARK), ("pressed", self.COLOR_ACCENT_DARK)],
            foreground=[("disabled", "#dce8f5")],
        )
        style.configure(
            "Camera.Treeview",
            rowheight=30,
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=self.COLOR_TEXT,
            bordercolor="#dbe3ed",
        )
        style.map(
            "Camera.Treeview",
            background=[("selected", self.COLOR_ACCENT)],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("Camera.Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Muted.TLabel", background=self.COLOR_PANEL, foreground=self.COLOR_MUTED)
        style.configure(
            "Title.TLabel",
            background=self.COLOR_PANEL,
            foreground=self.COLOR_TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure("Section.TLabel", background=self.COLOR_PANEL, foreground=self.COLOR_MUTED)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, background=self.COLOR_HEADER, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = tk.Frame(header, background=self.COLOR_HEADER)
        title_box.pack(side="left", padx=22, pady=12)
        tk.Label(
            title_box,
            text="相机阵列布局工作台",
            background=self.COLOR_HEADER,
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="建立物理位置、同位置相机组与标定邻接关系",
            background=self.COLOR_HEADER,
            foreground="#aebdd0",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        header_actions = tk.Frame(header, background=self.COLOR_HEADER)
        header_actions.pack(side="right", padx=20)
        ttk.Button(header_actions, text="打开布局", style="Toolbar.TButton", command=self._open_layout).pack(side="left", padx=4)
        ttk.Button(header_actions, text="导出布局", style="Primary.TButton", command=self._save_layout_as).pack(side="left", padx=4)

        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(16, 10))
        toolbar.pack(fill="x", padx=12, pady=(12, 8))
        ttk.Label(toolbar, text="矩阵", style="Title.TLabel").pack(side="left", padx=(0, 10))
        ttk.Label(toolbar, text="行", style="Section.TLabel").pack(side="left")
        ttk.Spinbox(toolbar, from_=1, to=100, textvariable=self.rows_var, width=5).pack(side="left", padx=(4, 8))
        ttk.Label(toolbar, text="默认列", style="Section.TLabel").pack(side="left")
        ttk.Spinbox(toolbar, from_=1, to=100, textvariable=self.columns_var, width=5).pack(side="left", padx=(4, 8))
        ttk.Button(toolbar, text="创建 / 重设", style="Toolbar.TButton", command=self._reset_matrix).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=14)
        ttk.Button(toolbar, text="加载代表图片", style="Toolbar.TButton", command=self._load_files).pack(side="left", padx=3)
        ttk.Button(toolbar, text="加载相机目录", style="Toolbar.TButton", command=self._load_folder).pack(side="left", padx=3)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=14)
        ttk.Button(toolbar, text="撤销", style="Toolbar.TButton", command=self.undo).pack(side="left", padx=3)
        ttk.Button(toolbar, text="重做", style="Toolbar.TButton", command=self.redo).pack(side="left", padx=3)

        autofill = ttk.Frame(self.root, style="Panel.TFrame", padding=(16, 8))
        autofill.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(autofill, text="快速排列", style="Title.TLabel").pack(side="left", padx=(0, 10))
        ttk.Combobox(
            autofill,
            textvariable=self.fill_order_var,
            values=("自然逐行", "蛇形排列"),
            width=12,
            state="readonly",
        ).pack(side="left", padx=5)
        ttk.Button(autofill, text="填充未放置相机", style="Toolbar.TButton", command=self._auto_fill).pack(side="left", padx=3)
        ttk.Button(autofill, text="清空位置", style="Toolbar.TButton", command=self._clear_placements).pack(side="left", padx=3)
        ttk.Label(
            autofill,
            text="拖入已有位置会追加为同位置相机组，不会覆盖；右键位置可管理组内相机或设置物理空洞。",
            style="Muted.TLabel",
        ).pack(side="left", padx=16)

        self.main_paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        left = ttk.Frame(self.main_paned, style="Panel.TFrame", padding=12, width=390)
        right = ttk.Frame(self.main_paned, style="Panel.TFrame", padding=10)
        self.main_paned.add(left, weight=0)
        self.main_paned.add(right, weight=1)
        self.root.after_idle(self._set_initial_sash)
        self._build_camera_panel(left)
        self._build_grid_panel(right)

        status = tk.Frame(self.root, background=self.COLOR_HEADER, padx=14, pady=7)
        status.pack(fill="x")
        tk.Label(status, textvariable=self.status_var, background=self.COLOR_HEADER, foreground="#dbe7f4").pack(side="left", fill="x", expand=True)
        tk.Label(status, textvariable=self.summary_var, background=self.COLOR_HEADER, foreground="#9fb4cb").pack(side="right")

    def _set_initial_sash(self) -> None:
        try:
            self.main_paned.sashpos(0, 390)
        except tk.TclError:
            pass

    def _build_camera_panel(self, parent) -> None:
        ttk.Label(parent, text="相机资源", style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text="选择或拖动任意相机到右侧物理位置", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        search_row = ttk.Frame(parent)
        search_row.pack(fill="x", pady=(6, 4))
        ttk.Entry(search_row, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        ttk.Combobox(
            search_row,
            textvariable=self.filter_var,
            values=("全部", "未放置", "已放置"),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(4, 0))

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True)
        self.camera_tree = ttk.Treeview(
            tree_frame,
            columns=("state", "position", "source"),
            show="tree headings",
            style="Camera.Treeview",
            selectmode="browse",
        )
        self.camera_tree.heading("#0", text="相机 ID")
        self.camera_tree.heading("state", text="状态")
        self.camera_tree.heading("position", text="位置")
        self.camera_tree.heading("source", text="来源")
        self.camera_tree.column("#0", width=160, minwidth=110)
        self.camera_tree.column("state", width=56, anchor="center", stretch=False)
        self.camera_tree.column("position", width=54, anchor="center", stretch=False)
        self.camera_tree.column("source", width=82, minwidth=60)
        self.camera_tree.tag_configure("unplaced", foreground="#a55b16")
        self.camera_tree.tag_configure("placed", foreground="#236b4b")
        self.camera_tree.tag_configure("multi", foreground="#6941a5")
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.camera_tree.yview)
        self.camera_tree.configure(yscrollcommand=tree_scroll.set)
        self.camera_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        preview_box = ttk.LabelFrame(parent, text="选中相机", style="Panel.TLabelframe", padding=8)
        preview_box.pack(fill="x", pady=(10, 4))
        self.preview_label = ttk.Label(preview_box, text="尚未选择", anchor="center", compound="top")
        self.preview_label.pack(fill="both", expand=True)

        camera_buttons = ttk.Frame(parent)
        camera_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(camera_buttons, text="重命名", style="Toolbar.TButton", command=self._rename_selected_camera).pack(side="left", padx=2)
        ttk.Button(camera_buttons, text="删除", style="Toolbar.TButton", command=self._remove_selected_camera).pack(side="left", padx=2)
        ttk.Button(camera_buttons, text="定位", style="Toolbar.TButton", command=self._locate_selected_camera).pack(side="left", padx=2)

        self.search_var.trace_add("write", lambda *_: self.refresh_palette())
        self.filter_var.trace_add("write", lambda *_: self.refresh_palette())
        self.camera_tree.bind("<<TreeviewSelect>>", self._on_tree_selection)
        self.camera_tree.bind("<ButtonPress-1>", self._on_tree_press, add=True)
        self.camera_tree.bind("<B1-Motion>", self._on_drag_motion, add=True)
        self.camera_tree.bind("<ButtonRelease-1>", self._on_tree_release, add=True)
        self.camera_tree.bind("<Double-1>", lambda _event: self._locate_selected_camera())

    def _build_grid_panel(self, parent) -> None:
        settings = ttk.Frame(parent, style="Panel.TFrame", padding=(4, 2, 4, 8))
        settings.pack(fill="x", pady=(0, 4))
        ttk.Label(settings, text="物理位置矩阵", style="Title.TLabel").pack(side="left", padx=(0, 16))
        ttk.Label(settings, text="邻接", style="Section.TLabel").pack(side="left")
        ttk.Combobox(
            settings,
            textvariable=self.neighbour_mode_var,
            values=("8 邻域", "4 邻域"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 8))
        ttk.Label(settings, text="半径").pack(side="left")
        ttk.Spinbox(settings, from_=1, to=8, textvariable=self.neighbour_radius_var, width=4).pack(side="left", padx=4)
        ttk.Button(settings, text="检查布局", style="Toolbar.TButton", command=self._show_validation).pack(side="left", padx=5)
        ttk.Label(settings, text="卡片大小", style="Section.TLabel").pack(side="left", padx=(18, 2))
        ttk.Scale(
            settings,
            from_=110,
            to=220,
            variable=self.cell_size_var,
            command=lambda _value: self._schedule_grid_refresh(),
        ).pack(side="left", fill="x", expand=True, padx=4)

        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True)
        self.grid_canvas = tk.Canvas(holder, background=self.COLOR_BG, highlightthickness=0)
        vertical = ttk.Scrollbar(holder, orient="vertical", command=self.grid_canvas.yview)
        horizontal = ttk.Scrollbar(holder, orient="horizontal", command=self.grid_canvas.xview)
        self.grid_canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        self.grid_inner = tk.Frame(self.grid_canvas, background=self.COLOR_BG)
        self.grid_window = self.grid_canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind("<Configure>", self._update_scroll_region)
        self.grid_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.grid_canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.grid_canvas.bind("<Configure>", lambda _event: self._update_scroll_region())

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())
        self.root.bind_all("<Control-s>", lambda _event: self._save_layout())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self._save_layout_as())
        self.root.bind_all("<Control-o>", lambda _event: self._open_layout())
        self.root.bind_all("<Escape>", self._on_escape)

    def _on_escape(self, _event=None) -> None:
        self._cancel_drag()
        self._clear_selection()
        self.status_var.set("已取消当前操作")

    def _checkpoint(self) -> None:
        self.undo_stack.append(self.model.to_dict())
        if len(self.undo_stack) > self.HISTORY_LIMIT:
            del self.undo_stack[0]
        self.redo_stack.clear()

    def _mutate(self, callback, success_message: str | None = None) -> bool:
        self._checkpoint()
        try:
            callback()
        except (LayoutError, OSError) as exc:
            self.undo_stack.pop()
            messagebox.showerror("操作失败", str(exc), parent=self.root)
            return False
        self.dirty = True
        self.refresh_all()
        if success_message:
            self.status_var.set(success_message)
        return True

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("没有可以撤销的操作")
            return
        self.redo_stack.append(self.model.to_dict())
        self.model = CameraGridLayout.from_dict(self.undo_stack.pop())
        self.dirty = True
        self._clear_selection()
        self.refresh_all()
        self.status_var.set("已撤销")

    def redo(self) -> None:
        if not self.redo_stack:
            self.status_var.set("没有可以重做的操作")
            return
        self.undo_stack.append(self.model.to_dict())
        self.model = CameraGridLayout.from_dict(self.redo_stack.pop())
        self.dirty = True
        self._clear_selection()
        self.refresh_all()
        self.status_var.set("已重做")

    def refresh_all(self) -> None:
        self.rows_var.set(self.model.row_count)
        self.columns_var.set(self.model.default_columns)
        self.refresh_palette()
        self.refresh_grid()
        self._refresh_summary()
        self.root.title(f"相机矩阵排列工具{' *' if self.dirty else ''}")

    def _refresh_summary(self) -> None:
        report = self.model.validate(self._neighbour_mode(), self._neighbour_radius())
        self.summary_var.set(
            f"相机 {report.loaded}  |  已放置 {report.placed}  |  物理位置 {report.occupied_positions}"
            f"  |  同位置组 {report.multi_camera_positions}  |  邻接对 {report.pair_count}"
        )

    def _on_neighbour_settings_changed(self, *_args) -> None:
        if self._suppress_setting_change:
            return
        self.dirty = True
        self._refresh_summary()
        self.root.title("相机矩阵排列工具 *")

    def refresh_palette(self) -> None:
        previous = self.selected_camera_id
        self.camera_tree.delete(*self.camera_tree.get_children())
        query = self.search_var.get().strip().lower()
        filter_mode = self.filter_var.get()
        placed = self.model.placed_camera_ids()
        for camera_id in sorted(self.model.cameras, key=natural_key):
            item = self.model.cameras[camera_id]
            is_placed = camera_id in placed
            if filter_mode == "未放置" and is_placed:
                continue
            if filter_mode == "已放置" and not is_placed:
                continue
            searchable = f"{camera_id} {item.preview_path} {item.source_path}".lower()
            if query and query not in searchable:
                continue
            source = Path(item.source_path).name or item.source_kind
            position = self.model.position_of(camera_id)
            if position is None:
                state, position_text, tag = "待放", "—", "unplaced"
            else:
                slot = self.model.rows[position[0]][position[1]]
                state = "同位" if len(slot.camera_ids) > 1 else "已放"
                position_text = f"{position[0] + 1},{position[1] + 1}"
                tag = "multi" if len(slot.camera_ids) > 1 else "placed"
            self.camera_tree.insert(
                "",
                "end",
                iid=camera_id,
                text=camera_id,
                values=(state, position_text, source),
                tags=(tag,),
            )
        if previous and self.camera_tree.exists(previous):
            self.camera_tree.selection_set(previous)
            self.camera_tree.see(previous)
        elif previous:
            self.selected_camera_id = None
        self._update_selected_preview()

    def refresh_grid(self) -> None:
        for child in self.grid_inner.winfo_children():
            child.destroy()
        self.slot_widgets.clear()
        self.slot_base_borders.clear()
        size = max(110, int(self.cell_size_var.get()))
        image_height = max(48, int(size * 0.48))
        for row_index, row in enumerate(self.model.rows):
            controls = tk.Frame(self.grid_inner, background="#dfe7f0", padx=7, pady=7)
            controls.grid(row=row_index, column=0, padx=(6, 8), pady=6, sticky="ns")
            tk.Label(
                controls,
                text=f"ROW {row_index + 1:02d}",
                background="#dfe7f0",
                foreground=self.COLOR_TEXT,
                font=("Microsoft YaHei UI", 9, "bold"),
            ).pack()
            tk.Label(
                controls,
                text=f"{len(row)} 个位置",
                background="#dfe7f0",
                foreground=self.COLOR_MUTED,
            ).pack(pady=(3, 7))
            buttons = tk.Frame(controls, background="#dfe7f0")
            buttons.pack()
            ttk.Button(buttons, text="−", width=2, command=lambda r=row_index: self._change_row_length(r, -1)).pack(side="left")
            ttk.Button(buttons, text="+", width=2, command=lambda r=row_index: self._change_row_length(r, 1)).pack(side="left", padx=(2, 0))

            for column_index, slot in enumerate(row):
                if not slot.enabled:
                    background, border = "#d9e0e8", "#aab4c0"
                elif len(slot.camera_ids) > 1:
                    background, border = "#ffffff", self.COLOR_MULTI
                elif slot.camera_ids:
                    background, border = "#ffffff", self.COLOR_ACCENT
                else:
                    background, border = "#f9fbfd", "#c4cfdb"
                frame = tk.Frame(
                    self.grid_inner,
                    width=size,
                    height=size,
                    background=background,
                    highlightthickness=2,
                    highlightbackground=border,
                    cursor="hand2" if slot.enabled else "arrow",
                )
                frame.grid(row=row_index, column=column_index + 1, padx=6, pady=6)
                frame.grid_propagate(False)
                frame._slot_pos = (row_index, column_index)  # type: ignore[attr-defined]
                self.slot_widgets[(row_index, column_index)] = frame
                self.slot_base_borders[(row_index, column_index)] = border

                header_color = (
                    "#c8d0da"
                    if not slot.enabled
                    else self.COLOR_MULTI
                    if len(slot.camera_ids) > 1
                    else self.COLOR_ACCENT
                    if slot.camera_ids
                    else "#e8eef5"
                )
                header_foreground = "#ffffff" if slot.enabled and slot.camera_ids else self.COLOR_MUTED
                card_header = tk.Frame(frame, background=header_color, height=25)
                card_header.pack(fill="x")
                card_header.pack_propagate(False)
                coordinate = tk.Label(
                    card_header,
                    text=f"R{row_index + 1:02d} · C{column_index + 1:02d}",
                    background=header_color,
                    foreground=header_foreground,
                    font=("Microsoft YaHei UI", 8, "bold"),
                )
                coordinate.pack(side="left", padx=6)
                count_text = "空洞" if not slot.enabled else f"{len(slot.camera_ids)} 台" if slot.camera_ids else "空位置"
                count = tk.Label(
                    card_header,
                    text=count_text,
                    background=header_color,
                    foreground=header_foreground,
                    font=("Microsoft YaHei UI", 8),
                )
                count.pack(side="right", padx=6)

                if not slot.enabled:
                    body = tk.Label(
                        frame,
                        text="此处没有相机位\n右键可恢复",
                        background=background,
                        foreground="#7d8897",
                        font=("Microsoft YaHei UI", 9),
                    )
                    body.pack(fill="both", expand=True)
                    interactive_widgets = (frame, card_header, coordinate, count, body)
                elif slot.camera_ids:
                    primary_id = (
                        self.selected_camera_id
                        if self.selected_camera_id in slot.camera_ids
                        else slot.camera_ids[0]
                    )
                    assert primary_id is not None
                    item = self.model.cameras[primary_id]
                    photo = self.thumbnail_cache.get(item.preview_path, size - 10, image_height)
                    preview = tk.Label(
                        frame,
                        image=photo,
                        background=background,
                        cursor="fleur",
                    )
                    preview.image = photo
                    preview.pack(fill="x", padx=4, pady=(4, 2))
                    preview._camera_id = primary_id  # type: ignore[attr-defined]
                    names = "\n".join(slot.camera_ids[:2])
                    if len(slot.camera_ids) > 2:
                        names += f"\n另有 {len(slot.camera_ids) - 2} 台…"
                    name_label = tk.Label(
                        frame,
                        text=names,
                        background=background,
                        foreground=self.COLOR_TEXT,
                        font=("Microsoft YaHei UI", 8, "bold" if len(slot.camera_ids) > 1 else "normal"),
                        wraplength=size - 12,
                        justify="center",
                    )
                    name_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))
                    name_label._camera_id = primary_id  # type: ignore[attr-defined]
                    interactive_widgets = (frame, card_header, coordinate, count, preview, name_label)
                else:
                    body = tk.Label(
                        frame,
                        text="＋\n拖入相机\n可添加多台",
                        background=background,
                        foreground="#8290a2",
                        font=("Microsoft YaHei UI", 10),
                    )
                    body.pack(fill="both", expand=True)
                    interactive_widgets = (frame, card_header, coordinate, count, body)
                for widget in interactive_widgets:
                    widget._slot_pos = (row_index, column_index)  # type: ignore[attr-defined]
                    self._bind_slot_widget(widget, row_index, column_index)
        self._update_scroll_region()

    def _bind_slot_widget(self, widget, row: int, column: int) -> None:
        widget.bind("<ButtonPress-1>", lambda event, r=row, c=column: self._on_slot_press(event, r, c))
        widget.bind("<B1-Motion>", self._on_drag_motion)
        widget.bind("<ButtonRelease-1>", lambda event, r=row, c=column: self._on_slot_release(event, r, c))
        widget.bind("<Button-3>", lambda event, r=row, c=column: self._show_slot_menu(event, r, c))
        widget.bind("<Enter>", lambda _event, r=row, c=column: self._on_slot_hover(r, c, True))
        widget.bind("<Leave>", lambda _event, r=row, c=column: self._on_slot_hover(r, c, False))

    def _on_slot_hover(self, row: int, column: int, entered: bool) -> None:
        position = (row, column)
        if self.drag_target is not None:
            return
        frame = self.slot_widgets.get(position)
        if frame is not None:
            frame.configure(
                highlightbackground=self.COLOR_ACCENT_DARK if entered else self.slot_base_borders[position],
                highlightthickness=3 if entered else 2,
            )

    def _on_slot_press(self, event, row: int, column: int) -> str:
        self.current_slot = (row, column)
        slot = self.model.rows[row][column]
        camera_id = getattr(event.widget, "_camera_id", None)
        if camera_id not in slot.camera_ids:
            camera_id = self.selected_camera_id if self.selected_camera_id in slot.camera_ids else slot.camera_id
        self.drag_payload = ("camera", camera_id, (row, column)) if camera_id else None
        self.drag_start = (event.x_root, event.y_root)
        if camera_id:
            self._select_camera(camera_id)
            self.status_var.set(f"拖动 {camera_id} 可移动到空位置，或加入另一个同位置相机组")
        return "break"

    def _on_slot_release(self, event, row: int, column: int) -> str:
        target = self._slot_under_pointer(event.x_root, event.y_root)
        payload = self.drag_payload
        start = self.drag_start
        moved = bool(start and abs(event.x_root - start[0]) + abs(event.y_root - start[1]) > 5)
        self._finish_drag_feedback()
        self.drag_payload = None
        self.drag_start = None
        if moved and payload and payload[1] and target:
            self._place_camera(payload[1], target[0], target[1])
        elif not moved and self.model.rows[row][column].enabled:
            slot_camera = getattr(event.widget, "_camera_id", None) or self.model.rows[row][column].camera_id
            if slot_camera:
                self._select_camera(slot_camera)
            elif self.selected_camera_id:
                self._place_camera(self.selected_camera_id, row, column)
        return "break"

    def _on_tree_press(self, event) -> None:
        item_id = self.camera_tree.identify_row(event.y)
        if not item_id:
            self.drag_payload = None
            return
        self.camera_tree.selection_set(item_id)
        self.selected_camera_id = item_id
        self.drag_payload = ("camera", item_id, self.model.position_of(item_id))
        self.drag_start = (event.x_root, event.y_root)
        self._update_selected_preview()

    def _on_drag_motion(self, event) -> str:
        if not self.drag_payload or not self.drag_payload[1] or not self.drag_start:
            return "break"
        distance = abs(event.x_root - self.drag_start[0]) + abs(event.y_root - self.drag_start[1])
        if distance <= 5:
            return "break"
        camera_id = self.drag_payload[1]
        if self.drag_window is None:
            self.drag_window = tk.Toplevel(self.root)
            self.drag_window.overrideredirect(True)
            self.drag_window.attributes("-topmost", True)
            try:
                self.drag_window.attributes("-alpha", 0.94)
            except tk.TclError:
                pass
            ghost = tk.Frame(
                self.drag_window,
                background=self.COLOR_HEADER,
                highlightbackground=self.COLOR_ACCENT,
                highlightthickness=2,
                padx=7,
                pady=6,
            )
            ghost.pack()
            item = self.model.cameras[camera_id]
            self.drag_photo = self.thumbnail_cache.get(item.preview_path, 86, 52)
            tk.Label(ghost, image=self.drag_photo, background=self.COLOR_HEADER).pack(side="left")
            tk.Label(
                ghost,
                text=f"移动相机\n{camera_id}",
                background=self.COLOR_HEADER,
                foreground="#ffffff",
                justify="left",
                font=("Microsoft YaHei UI", 9, "bold"),
            ).pack(side="left", padx=(8, 2))
            self.root.configure(cursor="fleur")
        self.drag_window.geometry(f"+{event.x_root + 18}+{event.y_root + 20}")
        self._set_drag_target(self._slot_under_pointer(event.x_root, event.y_root), camera_id)
        return "break"

    def _on_tree_release(self, event) -> None:
        payload = self.drag_payload
        target = self._slot_under_pointer(event.x_root, event.y_root)
        start = self.drag_start
        moved = bool(start and abs(event.x_root - start[0]) + abs(event.y_root - start[1]) > 5)
        self._finish_drag_feedback()
        self.drag_payload = None
        self.drag_start = None
        if moved and payload and payload[0] == "camera" and target:
            self._place_camera(payload[1], target[0], target[1])

    def _set_drag_target(self, target: tuple[int, int] | None, camera_id: str) -> None:
        if self.drag_target == target:
            return
        if self.drag_target in self.slot_widgets:
            previous = self.slot_widgets[self.drag_target]
            previous.configure(
                highlightbackground=self.slot_base_borders[self.drag_target],
                highlightthickness=2,
            )
        self.drag_target = target
        if target not in self.slot_widgets:
            self.status_var.set(f"正在移动 {camera_id}；请拖到一个物理位置卡片上")
            return
        slot = self.model.rows[target[0]][target[1]]
        frame = self.slot_widgets[target]
        if not slot.enabled:
            frame.configure(highlightbackground="#d64545", highlightthickness=4)
            self.status_var.set("该位置是物理空洞，无法放置")
        else:
            frame.configure(highlightbackground=self.COLOR_SUCCESS, highlightthickness=4)
            if camera_id in slot.camera_ids:
                self.status_var.set(f"{camera_id} 已经位于 R{target[0] + 1} C{target[1] + 1}")
            elif slot.camera_ids:
                self.status_var.set(
                    f"松开后加入 R{target[0] + 1} C{target[1] + 1}，形成 {len(slot.camera_ids) + 1} 台同位置相机组"
                )
            else:
                self.status_var.set(f"松开后放入 R{target[0] + 1} C{target[1] + 1}")

    def _finish_drag_feedback(self) -> None:
        if self.drag_target in self.slot_widgets:
            frame = self.slot_widgets[self.drag_target]
            frame.configure(
                highlightbackground=self.slot_base_borders[self.drag_target],
                highlightthickness=2,
            )
        self.drag_target = None
        if self.drag_window is not None:
            try:
                self.drag_window.destroy()
            except tk.TclError:
                pass
        self.drag_window = None
        self.drag_photo = None
        try:
            self.root.configure(cursor="")
        except tk.TclError:
            pass

    def _cancel_drag(self) -> None:
        self._finish_drag_feedback()
        self.drag_payload = None
        self.drag_start = None

    def _slot_under_pointer(self, x_root: int, y_root: int) -> tuple[int, int] | None:
        widget = self.root.winfo_containing(x_root, y_root)
        while widget is not None:
            position = getattr(widget, "_slot_pos", None)
            if position is not None:
                return position
            widget = getattr(widget, "master", None)
        return None

    def _show_slot_menu(self, event, row: int, column: int) -> str:
        self.current_slot = (row, column)
        slot = self.model.rows[row][column]
        menu = tk.Menu(self.root, tearoff=False)
        if slot.enabled and self.selected_camera_id and self.selected_camera_id not in slot.camera_ids:
            menu.add_command(
                label=f"把已选相机加入此位置",
                command=lambda camera_id=self.selected_camera_id: self._place_camera(camera_id, row, column),
            )
            menu.add_separator()
        if slot.camera_ids:
            select_menu = tk.Menu(menu, tearoff=False)
            remove_menu = tk.Menu(menu, tearoff=False)
            for camera_id in slot.camera_ids:
                select_menu.add_command(label=camera_id, command=lambda c=camera_id: self._select_camera(c))
                remove_menu.add_command(label=camera_id, command=lambda c=camera_id: self._remove_camera_from_position(c))
            menu.add_cascade(label=f"选择组内相机（{len(slot.camera_ids)} 台）", menu=select_menu)
            menu.add_cascade(label="从此位置移除一台", menu=remove_menu)
            menu.add_command(label="清空此位置全部相机", command=lambda: self._clear_slot(row, column))
            menu.add_separator()
        menu.add_command(
            label="设为物理空洞" if slot.enabled else "恢复为可用格子",
            command=lambda: self._toggle_gap(row, column),
        )
        menu.add_separator()
        menu.add_command(label="在前面插入格子", command=lambda: self._insert_slot(row, column))
        menu.add_command(label="在后面插入格子", command=lambda: self._insert_slot(row, column + 1))
        menu.add_command(label="删除这个格子", command=lambda: self._remove_slot(row, column))
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _place_camera(self, camera_id: str, row: int, column: int) -> None:
        slot = self.model.rows[row][column]
        if not slot.enabled:
            self.status_var.set("目标是物理空洞，请先右键恢复")
            return
        if camera_id in slot.camera_ids:
            self.status_var.set(f"{camera_id} 已在第 {row + 1} 行第 {column + 1} 列")
            return
        target_count = len(slot.camera_ids)
        message = (
            f"已将 {camera_id} 加入第 {row + 1} 行第 {column + 1} 列（同位置共 {target_count + 1} 台）"
            if target_count
            else f"已将 {camera_id} 放到第 {row + 1} 行第 {column + 1} 列"
        )
        self._mutate(lambda: self.model.place_camera(camera_id, row, column), message)

    def _clear_slot(self, row: int, column: int) -> None:
        camera_ids = list(self.model.rows[row][column].camera_ids)
        if not camera_ids:
            return
        if len(camera_ids) > 1 and not messagebox.askyesno(
            "清空位置",
            f"此位置有 {len(camera_ids)} 台相机，全部移回未放置列表吗？",
            parent=self.root,
        ):
            return
        self._mutate(
            lambda: self.model.clear_slot(row, column),
            f"已从此位置移除 {len(camera_ids)} 台相机，原图仍保留",
        )

    def _remove_camera_from_position(self, camera_id: str) -> None:
        position = self.model.position_of(camera_id)
        if position is None:
            return
        self._mutate(
            lambda: self.model.remove_camera_from_slot(camera_id),
            f"已将 {camera_id} 移回未放置列表",
        )

    def _toggle_gap(self, row: int, column: int) -> None:
        slot = self.model.rows[row][column]
        if slot.camera_ids and not messagebox.askyesno(
            "设为空洞", f"此位置的 {len(slot.camera_ids)} 台相机会回到未放置列表，继续吗？", parent=self.root
        ):
            return
        self._mutate(lambda: self.model.toggle_slot_enabled(row, column))

    def _insert_slot(self, row: int, column: int) -> None:
        self._mutate(lambda: self.model.insert_slot(row, column), f"第 {row + 1} 行已增加一个位置")

    def _remove_slot(self, row: int, column: int) -> None:
        slot = self.model.rows[row][column]
        if slot.camera_ids and not messagebox.askyesno(
            "删除位置", f"此位置的 {len(slot.camera_ids)} 台相机会回到未放置列表，继续吗？", parent=self.root
        ):
            return
        self._mutate(lambda: self.model.remove_slot(row, column), f"第 {row + 1} 行已删除一个位置")

    def _change_row_length(self, row: int, delta: int) -> None:
        old_length = len(self.model.rows[row])
        new_length = old_length + delta
        if new_length <= 0:
            self.status_var.set("每行至少保留一个位置")
            return
        if delta < 0 and self.model.rows[row][-1].camera_ids:
            camera_count = len(self.model.rows[row][-1].camera_ids)
            if not messagebox.askyesno(
                "减少位置", f"最后一个位置中的 {camera_count} 台相机会回到未放置列表，继续吗？", parent=self.root
            ):
                return
        self._mutate(lambda: self.model.set_row_length(row, new_length), f"第 {row + 1} 行现在有 {new_length} 个格子")

    def _reset_matrix(self) -> None:
        try:
            rows, columns = int(self.rows_var.get()), int(self.columns_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("参数错误", "行数和列数必须是正整数", parent=self.root)
            return
        if rows <= 0 or columns <= 0:
            messagebox.showerror("参数错误", "行数和列数必须是正整数", parent=self.root)
            return
        if self.model.placed_camera_ids() and not messagebox.askyesno(
            "重设矩阵", "当前格子中的相机会回到未放置列表，已加载图片不会删除。继续吗？", parent=self.root
        ):
            return
        self._mutate(lambda: self.model.reset_grid(rows, columns), f"已创建 {rows} 行 × {columns} 列矩阵")

    def _auto_fill(self) -> None:
        order = "serpentine" if self.fill_order_var.get() == "蛇形排列" else "row_major"
        remaining: list[str] = []

        def fill() -> None:
            nonlocal remaining
            remaining = self.model.auto_fill(order)

        if self._mutate(fill):
            if remaining:
                messagebox.showwarning(
                    "格子不足", f"还有 {len(remaining)} 台相机未放置，请增加格子或减少物理空洞。", parent=self.root
                )
            else:
                self.status_var.set("已按自然顺序填充所有可用空格")

    def _clear_placements(self) -> None:
        if not self.model.placed_camera_ids():
            return
        if not messagebox.askyesno("清空布局", "清空所有格子中的相机吗？图片仍会保留在左侧列表。", parent=self.root):
            return

        def clear() -> None:
            for row in self.model.rows:
                for slot in row:
                    slot.camera_ids.clear()

        self._mutate(clear, "已清空格中相机")

    def _load_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="选择每台相机的代表图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("所有文件", "*.*")],
        )
        if not paths:
            return
        self._add_camera_items(camera_items_from_files(Path(path) for path in paths))

    def _load_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self.root, title="选择相机目录根目录或单帧图片目录")
        if not folder:
            return
        try:
            items = discover_camera_items(Path(folder))
        except LayoutError as exc:
            messagebox.showerror("加载失败", str(exc), parent=self.root)
            return
        if not items:
            messagebox.showwarning("没有图片", "目录中没有找到支持的图片或相机子目录。", parent=self.root)
            return
        self._add_camera_items(items)

    def _add_camera_items(self, items: list[CameraItem]) -> None:
        if not items:
            return
        result: tuple[int, list[str]] = (0, [])

        def add() -> None:
            nonlocal result
            result = self.model.add_cameras(items)

        if self._mutate(add):
            added, collisions = result
            self.status_var.set(f"新增 {added} 台相机，当前共 {len(self.model.cameras)} 台")
            if collisions:
                messagebox.showwarning(
                    "相机 ID 重复",
                    "以下 ID 已存在，因此没有重复加载：\n" + "\n".join(collisions[:20]) + ("\n..." if len(collisions) > 20 else ""),
                    parent=self.root,
                )

    def _on_tree_selection(self, _event=None) -> None:
        selection = self.camera_tree.selection()
        self.selected_camera_id = selection[0] if selection else None
        self._update_selected_preview()

    def _select_camera(self, camera_id: str) -> None:
        self.selected_camera_id = camera_id
        if self.camera_tree.exists(camera_id):
            self.camera_tree.selection_set(camera_id)
            self.camera_tree.see(camera_id)
        self._update_selected_preview()

    def _clear_selection(self) -> None:
        self.selected_camera_id = None
        self.current_slot = None
        self.camera_tree.selection_remove(*self.camera_tree.selection())
        self._update_selected_preview()

    def _update_selected_preview(self) -> None:
        camera_id = self.selected_camera_id
        if not camera_id or camera_id not in self.model.cameras:
            self.preview_label.configure(text="尚未选择", image="")
            self.preview_label.image = None
            return
        item = self.model.cameras[camera_id]
        photo = self.thumbnail_cache.get(item.preview_path, 285, 175)
        path_text = f"{camera_id}\n{Path(item.preview_path).name}"
        position = self.model.position_of(camera_id)
        if position is not None:
            slot = self.model.rows[position[0]][position[1]]
            path_text += f"\n位置 R{position[0] + 1} C{position[1] + 1} · 同位置 {len(slot.camera_ids)} 台"
        else:
            path_text += "\n尚未放置"
        if not Path(item.preview_path).is_file():
            path_text += "\n（预览文件不存在）"
        self.preview_label.configure(text=path_text, image=photo or "", compound="top")
        self.preview_label.image = photo

    def _rename_selected_camera(self) -> None:
        camera_id = self.selected_camera_id
        if not camera_id:
            self.status_var.set("请先选择一台相机")
            return
        new_id = simpledialog.askstring("重命名相机", "新的相机 ID：", initialvalue=camera_id, parent=self.root)
        if new_id is None or new_id.strip() == camera_id:
            return
        if self._mutate(lambda: self.model.rename_camera(camera_id, new_id.strip())):
            self.selected_camera_id = new_id.strip()
            self.refresh_all()

    def _remove_selected_camera(self) -> None:
        camera_id = self.selected_camera_id
        if not camera_id:
            self.status_var.set("请先选择一台相机")
            return
        if not messagebox.askyesno("删除相机", f"从项目中删除 {camera_id}？原始图片不会被删除。", parent=self.root):
            return
        if self._mutate(lambda: self.model.remove_camera(camera_id), f"已从项目移除 {camera_id}"):
            self.selected_camera_id = None

    def _locate_selected_camera(self) -> None:
        camera_id = self.selected_camera_id
        if not camera_id:
            self.status_var.set("请先选择一台相机")
            return
        position = self.model.position_of(camera_id)
        if position is None:
            self.status_var.set(f"{camera_id} 尚未放置")
            return
        row, column = position
        self.grid_canvas.update_idletasks()
        bbox = self.grid_canvas.bbox("all")
        if bbox:
            total_width = max(1, bbox[2] - bbox[0])
            total_height = max(1, bbox[3] - bbox[1])
            size = max(110, int(self.cell_size_var.get())) + 12
            self.grid_canvas.xview_moveto(max(0.0, min(1.0, (column * size) / total_width)))
            self.grid_canvas.yview_moveto(max(0.0, min(1.0, (row * size) / total_height)))
        self.status_var.set(f"{camera_id} 位于第 {row + 1} 行第 {column + 1} 列")

    def _neighbour_mode(self) -> Literal[4, 8]:
        return 4 if self.neighbour_mode_var.get().startswith("4") else 8

    def _neighbour_radius(self) -> int:
        try:
            return max(1, int(self.neighbour_radius_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return 1

    def _show_validation(self) -> None:
        report = self.model.validate(self._neighbour_mode(), self._neighbour_radius())
        lines = [
            f"加载相机：{report.loaded}",
            f"已放置：{report.placed}",
            f"未放置：{report.unplaced}",
            f"已占用物理位置：{report.occupied_positions}",
            f"多相机位置：{report.multi_camera_positions}",
            f"可用空格：{report.active_empty_slots}",
            f"物理空洞：{report.disabled_slots}",
            f"邻接关系：{report.pair_count} 对",
            f"连通分量：{len(report.components)}（{'通过' if report.connected else '不连通'}）",
        ]
        if report.isolated:
            lines.append("孤立相机：" + ", ".join(report.isolated[:20]))
        if report.missing_previews:
            lines.append("预览文件缺失：" + ", ".join(report.missing_previews[:20]))
        if report.components and not report.connected:
            component_text = [f"分量 {index + 1}: {', '.join(component[:12])}" for index, component in enumerate(report.components)]
            lines.extend(component_text)
        messagebox.showinfo("布局检查", "\n".join(lines), parent=self.root)

    def _open_layout(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="打开相机矩阵布局",
            filetypes=[("相机矩阵 JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self._load_layout_path(Path(path), ask_discard=True)

    def _load_layout_path(self, path: Path, ask_discard: bool) -> None:
        if ask_discard and self.dirty and not messagebox.askyesno("打开布局", "当前未保存修改会丢失，继续吗？", parent=self.root):
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            layout = CameraGridLayout.from_dict(data)
        except (OSError, json.JSONDecodeError, LayoutError) as exc:
            messagebox.showerror("打开失败", str(exc), parent=self.root)
            return
        settings = data.get("neighbour_settings", {})
        if not isinstance(settings, dict):
            settings = {}
        self._suppress_setting_change = True
        try:
            if settings.get("mode") in {4, 8}:
                self.neighbour_mode_var.set(f"{settings['mode']} 邻域")
            if isinstance(settings.get("radius"), int) and settings["radius"] > 0:
                self.neighbour_radius_var.set(settings["radius"])
        finally:
            self._suppress_setting_change = False
        self.model = layout
        self.project_path = path.resolve()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self.thumbnail_cache.clear()
        self._clear_selection()
        self.refresh_all()
        self.status_var.set(f"已打开 {path}")

    def _save_layout(self) -> None:
        if self.project_path is None:
            self._save_layout_as()
        else:
            self._export_to(self.project_path)

    def _save_layout_as(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出相机矩阵布局",
            defaultextension=".json",
            filetypes=[("相机矩阵 JSON", "*.json")],
            initialfile=self.project_path.name if self.project_path else "camera_grid.json",
        )
        if path:
            self._export_to(Path(path))

    def _export_to(self, path: Path) -> None:
        report = self.model.validate(self._neighbour_mode(), self._neighbour_radius())
        warnings: list[str] = []
        if report.unplaced:
            warnings.append(f"还有 {report.unplaced} 台相机未放置")
        if report.placed > 1 and not report.connected:
            warnings.append(f"布局存在 {len(report.components)} 个不连通区域")
        if report.isolated:
            warnings.append(f"有 {len(report.isolated)} 台孤立相机")
        if warnings and not messagebox.askyesno(
            "布局尚不完整", "\n".join(warnings) + "\n\n仍然导出吗？", parent=self.root
        ):
            return
        try:
            outputs = export_layout(self.model, path, self._neighbour_mode(), self._neighbour_radius())
        except (OSError, LayoutError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.project_path = outputs[0].resolve()
        self.dirty = False
        self.refresh_all()
        self.status_var.set("已导出：" + ", ".join(output.name for output in outputs))

    def _schedule_grid_refresh(self) -> None:
        if self._grid_refresh_job is not None:
            try:
                self.root.after_cancel(self._grid_refresh_job)
            except tk.TclError:
                pass

        def refresh() -> None:
            self._grid_refresh_job = None
            self.refresh_grid()

        self._grid_refresh_job = self.root.after(120, refresh)

    def _update_scroll_region(self, _event=None) -> None:
        self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all"))

    def _on_mousewheel(self, event) -> str:
        self.grid_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _on_shift_mousewheel(self, event) -> str:
        self.grid_canvas.xview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _on_close(self) -> None:
        if self.dirty and not messagebox.askyesno("退出", "还有未保存修改，确定退出吗？", parent=self.root):
            return
        self._finish_drag_feedback()
        self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactively arrange camera images on a 2D matrix.")
    parser.add_argument("--rows", type=int, default=4, help="Initial number of grid rows.")
    parser.add_argument("--cols", type=int, default=4, help="Initial default number of grid columns.")
    parser.add_argument("--layout", type=Path, default=None, help="Open an existing camera-grid JSON file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if tk is None:
        raise RuntimeError("Tkinter is not available in this Python installation")
    root = tk.Tk()
    CameraGridApp(root, args.rows, args.cols, args.layout.resolve() if args.layout else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
