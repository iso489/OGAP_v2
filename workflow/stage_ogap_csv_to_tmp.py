#!/usr/bin/env python3
"""Stage OGAP CSV-referenced volumes into a Slurm local scratch directory.

The training code reads full 3D NIfTI volumes in DataLoader workers. On Rorqual,
network filesystem reads and the resulting Linux page cache can be charged
against the Slurm memory cgroup, which is what made the one-H100 jobs hit
Out Of Memory before sustained GPU work began. This helper copies the CSV's
path columns into $SLURM_TMPDIR, optionally decompressing .nii.gz to .nii, then
prints shell assignments pointing the sbatch script at rewritten staged CSVs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import re
import shutil
import shlex
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_COLUMNS = ("t1n", "t1c", "t2w", "t2f", "label")


def _log(message: str) -> None:
    print(f"[stage] {message}", file=sys.stderr, flush=True)


def _safe_assignment_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"unsafe assignment name: {name!r}")
    return name


def _drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = None
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
    except OSError as exc:
        _log(f"drop-file-cache skipped for {path}: {exc}")
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _unique_dest(src: Path, files_root: Path, decompress_gzip: bool) -> Path:
    resolved = src.resolve()
    # Deliberately key on the resolved path, not file contents: one Slurm job's
    # stage directory is transient, so stale content cannot survive past job end.
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
    name = src.name
    if decompress_gzip and name.endswith(".nii.gz"):
        name = name[:-3]
    return files_root / f"{digest}_{name}"


def _copy_or_decompress(src: Path, dest: Path, decompress_gzip: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        _drop_file_cache(dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        if decompress_gzip and src.name.endswith(".nii.gz"):
            with gzip.open(src, "rb") as in_f, open(tmp, "wb") as out_f:
                shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)
        else:
            shutil.copyfile(src, tmp)
        shutil.copystat(src, tmp, follow_symlinks=True)
        os.replace(tmp, dest)
        _drop_file_cache(src)
        _drop_file_cache(dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _csv_pairs(values: Iterable[str]) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"--csv expects NAME=/path/to/file.csv, got: {item!r}")
        name, path = item.split("=", 1)
        pairs.append((_safe_assignment_name(name), Path(path).expanduser()))
    return pairs


def _source_candidates(value: str, csv_path: Path) -> List[Path]:
    src = Path(value).expanduser()
    candidates = [src]
    if not src.is_absolute():
        csv_relative = (csv_path.parent / src).resolve()
        if csv_relative not in candidates:
            candidates.append(csv_relative)
    return candidates


def _resolve_source(value: str, csv_path: Path) -> Path:
    candidates = _source_candidates(value, csv_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _source_not_found_message(name: str, col: str, value: str, csv_path: Path) -> str:
    tried = ", ".join(str(p) for p in _source_candidates(value, csv_path))
    return f"{name}:{col} source not found for {value!r}; tried: {tried}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--csv", action="append", required=True, help="NAME=/path/to/input.csv")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decompress-gzip", action="store_true")
    parser.add_argument("--env-file", default=None, help="Write shell exports here and print the file path.")
    args = parser.parse_args()

    stage_root = Path(args.stage_root).expanduser()
    files_root = stage_root / "files"
    csv_root = stage_root / "csv"
    columns = tuple(col.strip() for col in args.columns.split(",") if col.strip())
    workers = max(1, int(args.workers))
    csv_specs = _csv_pairs(args.csv)

    stage_root.mkdir(parents=True, exist_ok=True)
    files_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Tuple[Path, List[str], List[Dict[str, str]]]] = {}
    path_map: Dict[Path, Path] = {}
    missing_columns: Dict[str, List[str]] = {}

    for name, csv_path in csv_specs:
        if not csv_path.exists():
            raise FileNotFoundError(f"{name} CSV not found: {csv_path}")
        fieldnames, rows = _read_csv(csv_path)
        loaded[name] = (csv_path, fieldnames, rows)
        missing_columns[name] = [col for col in columns if col not in fieldnames]
        for row in rows:
            for col in columns:
                value = (row.get(col) or "").strip()
                if not value:
                    continue
                src = _resolve_source(value, csv_path)
                if not src.exists():
                    raise FileNotFoundError(_source_not_found_message(name, col, value, csv_path))
                if src.is_file():
                    path_map[src.resolve()] = _unique_dest(src, files_root, args.decompress_gzip)

    if missing_columns:
        for name, cols in missing_columns.items():
            if cols:
                _log(f"{name}: columns absent and skipped: {','.join(cols)}")

    _log(
        f"staging {len(path_map)} unique files from {len(csv_specs)} CSV(s) "
        f"to {stage_root} with workers={workers}, decompress_gzip={int(args.decompress_gzip)}"
    )

    if path_map:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_copy_or_decompress, src, dest, args.decompress_gzip)
                for src, dest in path_map.items()
            ]
            completed = 0
            errors: List[Exception] = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
                completed += 1
                if completed == len(futures) or completed % 250 == 0:
                    _log(f"staged {completed}/{len(futures)} files")
            if errors:
                raise RuntimeError(
                    f"{len(errors)} staging error(s); first error: {errors[0]}"
                ) from errors[0]

    assignments: Dict[str, Path] = {}
    for name, (csv_path, fieldnames, rows) in loaded.items():
        staged_rows: List[Dict[str, str]] = []
        for row in rows:
            new_row = dict(row)
            for col in columns:
                value = (new_row.get(col) or "").strip()
                if not value:
                    continue
                src = _resolve_source(value, csv_path)
                resolved = src.resolve()
                if resolved in path_map:
                    new_row[col] = str(path_map[resolved])
            staged_rows.append(new_row)
        staged_csv = csv_root / f"{name.lower()}.csv"
        _write_csv(staged_csv, fieldnames, staged_rows)
        assignments[name] = staged_csv
        _log(f"{name}: {csv_path} -> {staged_csv} ({len(staged_rows)} rows)")

    env_lines = [f"export {name}={shlex.quote(str(path))}" for name, path in assignments.items()]
    if args.env_file:
        env_file = Path(args.env_file).expanduser()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = env_file.with_name(f".{env_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        os.replace(tmp, env_file)
        print(str(env_file))
    else:
        for line in env_lines:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
