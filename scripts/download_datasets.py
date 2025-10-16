#!/usr/bin/env python3
"""Download common time-series datasets into the local data/raw directory.

Usage examples:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset ett
    python scripts/download_datasets.py --all --force
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import shutil
import tempfile
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata.json"


@dataclass
class Resource:
    url: str
    save_as: str
    compression: Optional[str] = None
    keep_archive: bool = False


@dataclass
class Dataset:
    key: str
    display_name: str
    description: str
    frequency: Iterable[str]
    domain: str
    resources: List[Resource]
    license: str
    source: str


class DatasetRegistry:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path
        self._datasets: Dict[str, Dataset] = {}
        self._load()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with self.metadata_path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
        for key, info in payload.items():
            resources = [
                Resource(
                    url=res["url"],
                    save_as=res["save_as"],
                    compression=res.get("compression"),
                    keep_archive=res.get("keep_archive", False),
                )
                for res in info["resources"]
            ]
            dataset = Dataset(
                key=key,
                display_name=info["display_name"],
                description=info.get("description", ""),
                frequency=info.get("frequency", []),
                domain=info.get("domain", "unknown"),
                resources=resources,
                license=info.get("license", "unspecified"),
                source=info.get("source", "")
            )
            self._datasets[key] = dataset

    @property
    def keys(self) -> List[str]:
        return sorted(self._datasets.keys())

    def get(self, key: str) -> Dataset:
        try:
            return self._datasets[key]
        except KeyError as err:
            raise KeyError(f"Dataset '{key}' not found. Available: {', '.join(self.keys)}") from err

    def __contains__(self, key: str) -> bool:
        return key in self._datasets


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)


def human_readable_dataset(dataset: Dataset) -> str:
    freq = ", ".join(dataset.frequency) if dataset.frequency else "-"
    body = textwrap.dedent(
        f"""
        {dataset.display_name}
        Key: {dataset.key}
        Domain: {dataset.domain}
        Frequency: {freq}
        License: {dataset.license}
        Source: {dataset.source}
        Description: {dataset.description}
        Resources: {len(dataset.resources)} file(s)
        """
    ).strip()
    return body


def list_datasets(registry: DatasetRegistry) -> None:
    print("已收录的数据集：")
    for key in registry.keys:
        dataset = registry.get(key)
        print(f"- {key}: {dataset.display_name} ({dataset.domain}, {', '.join(dataset.frequency) if dataset.frequency else 'freq n/a'})")


def download_all(registry: DatasetRegistry, keys: Iterable[str], force: bool = False) -> None:
    ensure_directories()
    for key in keys:
        dataset = registry.get(key)
        download_dataset(dataset, force=force)


def download_dataset(dataset: Dataset, force: bool = False) -> None:
    dataset_dir = RAW_DIR / dataset.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ {dataset.display_name} ({dataset.key})")
    for resource in dataset.resources:
        target_path = dataset_dir / resource.save_as
        if target_path.exists() and not force and not resource.compression:
            print(f"  · 已存在文件 {target_path.name}，跳过 (use --force to redownload)")
            continue

        try:
            _download_resource(resource, target_path, force=force)
        except Exception as exc:  # broad catch to keep loop going
            print(f"  ! 下载 {resource.url} 失败: {exc}")
            continue


def _download_resource(resource: Resource, target_path: Path, force: bool = False) -> None:
    if resource.compression:
        archive_path = target_path if resource.keep_archive else target_path.with_suffix(target_path.suffix or ".tmp")
        if archive_path.exists() and not force:
            print(f"  · 已存在归档 {archive_path.name}，跳过解压 (use --force to redownload)")
        else:
            print(f"  · 下载 {resource.url} → {archive_path.name}")
            _stream_download(resource.url, archive_path, force=force)
        _extract_archive(archive_path, target_path.parent, resource.compression)
        if not resource.keep_archive and archive_path.exists():
            archive_path.unlink()
    else:
        if target_path.exists() and not force:
            print(f"  · 已存在文件 {target_path.name}，跳过 (use --force to redownload)")
            return
        print(f"  · 下载 {resource.url} → {target_path.name}")
        _stream_download(resource.url, target_path, force=force)


def _stream_download(url: str, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        return
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except urllib.error.URLError as err:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"无法下载 {url}: {err}") from err
    tmp_path.replace(output_path)


def _extract_archive(archive_path: Path, destination: Path, compression: str) -> None:
    compression = compression.lower()
    if compression == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
            print(f"  · 解压 {archive_path.name} → {destination}")
    elif compression in {"tar", "tar.gz", "tgz", "tar.bz2"}:
        mode = "r:"
        if compression == "tar.gz" or compression == "tgz":
            mode = "r:gz"
        elif compression == "tar.bz2":
            mode = "r:bz2"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(destination)
            print(f"  · 解压 {archive_path.name} → {destination}")
    else:
        raise ValueError(f"未知压缩格式: {compression}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download popular open-source time-series datasets.")
    parser.add_argument("--dataset", "-d", action="append", dest="datasets", help="指定单个或多个数据集键，可重复使用。")
    parser.add_argument("--all", action="store_true", help="下载全部数据集。")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集。")
    parser.add_argument("--info", metavar="KEY", help="输出指定数据集的详细信息。")
    parser.add_argument("--force", action="store_true", help="强制重新下载并覆盖现有文件。")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    registry = DatasetRegistry(METADATA_PATH)

    if args.list:
        list_datasets(registry)

    if args.info:
        key = args.info
        if key not in registry:
            print(f"未找到数据集 {key}，可选：{', '.join(registry.keys)}", file=sys.stderr)
            return 1
        dataset = registry.get(key)
        print(human_readable_dataset(dataset))

    targets: List[str] = []
    if args.all:
        targets.extend(registry.keys)
    if args.datasets:
        targets.extend(args.datasets)

    # Remove duplicates while maintaining order
    seen = set()
    ordered_targets = []
    for key in targets:
        if key not in registry:
            print(f"警告：未知数据集 {key}，已忽略。", file=sys.stderr)
            continue
        if key not in seen:
            seen.add(key)
            ordered_targets.append(key)

    if ordered_targets:
        download_all(registry, ordered_targets, force=args.force)
    elif not args.list and not args.info:
        print("未指定操作，使用 --list 查看可选数据集或 --dataset <key> 下载。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
