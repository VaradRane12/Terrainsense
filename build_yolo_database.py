from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AnnotationRow:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    line_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite database from a YOLO dataset.")
    parser.add_argument("--data-root", default="terrain_dataset", help="Dataset root containing train/valid/test folders.")
    parser.add_argument("--db-path", default="terrain_dataset.sqlite", help="Path to the SQLite database to create.")
    parser.add_argument("--stats-path", default="dataset_stats.json", help="Path to write dataset statistics JSON.")
    return parser.parse_args()


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def parse_label_file(label_path: Path, image_width: int, image_height: int) -> list[AnnotationRow]:
    annotations: list[AnnotationRow] = []
    for line_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label in {label_path}: line {line_index} has {len(parts)} fields")

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x_min = (x_center - width / 2.0) * image_width
        y_min = (y_center - height / 2.0) * image_height
        x_max = (x_center + width / 2.0) * image_width
        y_max = (y_center + height / 2.0) * image_height

        annotations.append(
            AnnotationRow(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                line_index=line_index,
            )
        )
    return annotations


def iter_dataset(data_root: Path) -> Iterable[tuple[str, Path, Path | None]]:
    for split in SPLITS:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"
        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.is_dir() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            yield split, image_path, label_path if label_path.exists() else None


def initialise_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS classes (
            class_id INTEGER PRIMARY KEY,
            class_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            file_name TEXT NOT NULL,
            image_path TEXT NOT NULL UNIQUE,
            label_path TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            has_label INTEGER NOT NULL,
            label_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            line_index INTEGER NOT NULL,
            x_center REAL NOT NULL,
            y_center REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            x_min REAL NOT NULL,
            y_min REAL NOT NULL,
            x_max REAL NOT NULL,
            y_max REAL NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def load_class_names(data_root: Path) -> list[str]:
    classes_file = data_root.parent / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    data_yaml = data_root / "data.yaml"
    if data_yaml.exists():
        for line in data_yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:"):
                raw_value = line.split("names:", 1)[1].strip()
                raw_value = raw_value.replace("'", '"')
                try:
                    names = json.loads(raw_value)
                except json.JSONDecodeError:
                    names = []
                return [str(name) for name in names]

    raise FileNotFoundError("Could not locate classes.txt or parse class names from data.yaml")


def build_database(data_root: Path, db_path: Path, stats_path: Path) -> dict[str, object]:
    class_names = load_class_names(data_root)
    split_image_counts = Counter()
    split_label_counts = Counter()
    split_annotation_counts = Counter()
    image_sizes: list[tuple[int, int]] = []
    class_annotation_counts = Counter()
    missing_labels: list[str] = []

    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    initialise_database(connection)

    connection.executemany(
        "INSERT OR REPLACE INTO classes (class_id, class_name) VALUES (?, ?)",
        [(index, class_name) for index, class_name in enumerate(class_names)],
    )

    image_rows = 0
    annotation_rows = 0

    for split, image_path, label_path in iter_dataset(data_root):
        with Image.open(image_path) as image:
            width, height = image.size

        file_size_bytes = image_path.stat().st_size
        has_label = int(label_path is not None)

        cursor = connection.execute(
            """
            INSERT INTO images (
                split, file_name, image_path, label_path, width, height,
                file_size_bytes, has_label, label_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                split,
                image_path.name,
                str(image_path),
                str(label_path) if label_path else None,
                width,
                height,
                file_size_bytes,
                has_label,
                0,
            ),
        )
        image_id = cursor.lastrowid
        image_rows += 1
        split_image_counts[split] += 1
        image_sizes.append((width, height))

        if label_path is None:
            missing_labels.append(str(image_path))
            continue

        annotations = parse_label_file(label_path, width, height)
        split_label_counts[split] += 1
        split_annotation_counts[split] += len(annotations)

        connection.executemany(
            """
            INSERT INTO annotations (
                image_id, class_id, line_index, x_center, y_center, width, height,
                x_min, y_min, x_max, y_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    image_id,
                    annotation.class_id,
                    annotation.line_index,
                    annotation.x_center,
                    annotation.y_center,
                    annotation.width,
                    annotation.height,
                    annotation.x_min,
                    annotation.y_min,
                    annotation.x_max,
                    annotation.y_max,
                )
                for annotation in annotations
            ],
        )
        annotation_rows += len(annotations)
        class_annotation_counts.update(annotation.class_id for annotation in annotations)

        connection.execute(
            "UPDATE images SET label_count = ? WHERE image_id = ?",
            (len(annotations), image_id),
        )

    connection.executemany(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        [
            ("dataset_root", str(data_root)),
            ("classes_file", str(data_root.parent / "classes.txt")),
            ("image_count", str(image_rows)),
            ("annotation_count", str(annotation_rows)),
            ("missing_label_count", str(len(missing_labels))),
        ],
    )
    connection.commit()

    width_values = [size[0] for size in image_sizes]
    height_values = [size[1] for size in image_sizes]

    stats = {
        "dataset_root": str(data_root),
        "database_path": str(db_path),
        "splits": {
            split: {
                "images": split_image_counts.get(split, 0),
                "label_files": split_label_counts.get(split, 0),
                "annotations": split_annotation_counts.get(split, 0),
            }
            for split in SPLITS
        },
        "classes": {str(class_id): class_names[class_id] for class_id in range(len(class_names))},
        "class_annotation_counts": {
            class_names[class_id]: class_annotation_counts.get(class_id, 0) for class_id in range(len(class_names))
        },
        "image_summary": {
            "count": image_rows,
            "missing_label_images": len(missing_labels),
            "width_min": min(width_values) if width_values else None,
            "width_max": max(width_values) if width_values else None,
            "height_min": min(height_values) if height_values else None,
            "height_max": max(height_values) if height_values else None,
        },
        "missing_label_images": missing_labels,
    }

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    connection.close()
    return stats


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    db_path = Path(args.db_path)
    stats_path = Path(args.stats_path)

    stats = build_database(data_root, db_path, stats_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()