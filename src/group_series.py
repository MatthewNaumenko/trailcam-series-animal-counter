#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Файл: group_series.py
Описание: Группировка JPG и MOV по сериям на основе времени съёмки.
         JPG: EXIF DateTimeOriginal (exifread)
         MOV: CreateDate/MediaCreateDate/DateTimeOriginal (exiftool)
         Серии формируются по времени съёмки; файлы без даты помещаются в NoDate.
Автор: naumenko33301 <naumenko33301@gmail.com>
Дата создания: 2025-10-20
Версия: 1.0
Лицензия: All Rights Reserved
"""

# Конфигурация
DEFAULT_ROOT = r"D:\xxxxx\xxxxxx"                      # Корневая папка с файлами
MINUTES     = 60                                       # Порог разрыва между файлами в серии (мин)
WORKERS     = None                                     # None=авто (4*CPU) для JPG; или число потоков
DRY_RUN     = False                                    # True=только план и лог, без перемещений
MOV_BATCH   = 500                                      # Размер батча для exiftool по MOV (оптимально 500–1500)

import argparse
import csv
import json
import os
import re
import sys
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# JPG EXIF
try:
    import exifread
except Exception:
    print("Нужна библиотека 'exifread'. Установите: pip install exifread", file=sys.stderr)
    raise

NODATE_DIRNAME = "NoDate"
SERIES_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_S\d{3}_\d{4}-\d{4}$")

EXT_JPG = {".jpg", ".jpeg", ".jpe", ".jfif", ".jif", ".pjpeg", ".pjp"}
EXT_MOV = {".mov"}  # По запросу - только MOV; при желании можно добавить .mp4 и т.п.

def _auto_workers():
    return max(8, (os.cpu_count() or 8) * 4)

def is_media_file(entry: os.DirEntry) -> bool:
    if not entry.is_file(follow_symlinks=False):
        return False
    ext = os.path.splitext(entry.name)[1].lower()
    return ext in EXT_JPG or ext in EXT_MOV

def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXT_JPG: return "jpg"
    if ext in EXT_MOV: return "mov"
    return "other"

def is_result_dir(path: Path) -> bool:
    n = path.name
    return n == NODATE_DIRNAME or SERIES_DIR_RE.match(n) is not None

def walk_media(root: Path):
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            p = Path(entry.path)
                            if is_result_dir(p):
                                continue
                            stack.append(p)
                        elif is_media_file(entry):
                            yield Path(entry.path)
                    except PermissionError:
                        print(f"Нет доступа: {entry.path}", file=sys.stderr)
        except PermissionError:
            print(f"Нет доступа: {cur}", file=sys.stderr)

# JPG: EXIF DateTimeOriginal
def parse_exif_datetime_jpg(path: Path):
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
        val = tags.get("EXIF DateTimeOriginal")
        if val:
            s = str(val)  # 'YYYY:MM:DD HH:MM:SS'
            try:
                dt = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
                return (path, dt, "")
            except Exception:
                return (path, None, f"Bad format: {s!r}")
        else:
            return (path, None, "No EXIF DateTimeOriginal")
    except Exception as e:
        return (path, None, f"Error: {e}")

# MOV: exiftool
def which_exiftool():
    # Ищем exiftool/exiftool.exe в PATH и рядом со скриптом
    from shutil import which as _which
    cand = _which("exiftool") or _which("exiftool.exe")
    if cand:
        return cand
    here = Path(__file__).resolve().parent
    local = here / "exiftool.exe"
    if local.exists():
        return str(local)
    return None

# Возвращает datetime без таймзоны
def parse_dt_string(dt_str: str):
    s = dt_str.strip()
    # Отрезаем тайм-зону, если есть
    if len(s) >= 19:
        s = s[:19]
    try:
        return datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

# Возвращает списки: dated: [(Path, datetime)], nodate: [(Path, reason)
def parse_mov_datetimes_exiftool(mov_paths, exiftool_path, batch_size=800):
    dated, nodate = [], []
    if not mov_paths:
        return dated, nodate
    if not exiftool_path:
        for p in mov_paths:
            nodate.append((p, "exiftool not found"))
        return dated, nodate

    for i in range(0, len(mov_paths), batch_size):
        chunk = mov_paths[i:i+batch_size]
        try:
            cmd = [
                exiftool_path,
                "-api", "largefilesupport=1",
                "-fast2",
                "-m",
                "-n",
                "-j",  
                "-DateTimeOriginal",
                "-CreateDate",
                "-MediaCreateDate",
            ] + [str(p) for p in chunk]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0 and not res.stdout.strip():
                for p in chunk:
                    reason = f"exiftool rc={res.returncode}: {res.stderr.strip()[:200]}"
                    nodate.append((p, reason))
                continue

            try:
                data = json.loads(res.stdout)
            except Exception as e:
                for p in chunk:
                    nodate.append((p, f"exiftool JSON parse error: {e}"))
                continue

            by_src = { Path(d.get("SourceFile")).resolve(): d for d in data if isinstance(d, dict) and d.get("SourceFile") }

            for p in chunk:
                rec = by_src.get(p.resolve())
                if not rec:
                    nodate.append((p, "no record from exiftool"))
                    continue
                if "Error" in rec and rec["Error"]:
                    nodate.append((p, f"exiftool: {rec['Error']}"))
                    continue
                dt_str = rec.get("DateTimeOriginal") or rec.get("CreateDate") or rec.get("MediaCreateDate")
                if not dt_str:
                    nodate.append((p, "No CreateDate/MediaCreateDate/DateTimeOriginal"))
                    continue
                dt = parse_dt_string(str(dt_str))
                if not dt:
                    nodate.append((p, f"Bad date format: {dt_str!r}"))
                else:
                    dated.append((p, dt))
        except Exception as e:
            for p in chunk:
                nodate.append((p, f"exiftool error: {e}"))
    return dated, nodate

# Общая логика
def ensure_unique_path(dest_dir: Path, base_name: str) -> Path:
    target = dest_dir / base_name
    if not target.exists():
        return target
    stem, ext = os.path.splitext(base_name)
    k = 1
    while True:
        cand = dest_dir / f"{stem}__{k}{ext}"
        if not cand.exists():
            return cand
        k += 1

def group_into_series(dated_items, minutes: int):
    by_date = {}
    for p, dt in dated_items:
        by_date.setdefault(dt.date(), []).append((p, dt))
    groups = []
    for d, lst in by_date.items():
        lst.sort(key=lambda x: x[1])
        series_idx = 1
        cur = [lst[0]]
        prev_dt = lst[0][1]
        start_dt = prev_dt
        for p, dt in lst[1:]:
            gap_minutes = (dt - prev_dt).total_seconds() / 60.0
            if dt.date() != d or gap_minutes > minutes:
                folder = f"{d:%Y-%m-%d}_S{series_idx:03d}_{start_dt:%H%M}-{prev_dt:%H%M}"
                groups.append((folder, list(cur)))
                series_idx += 1
                cur = [(p, dt)]
                start_dt = dt
            else:
                cur.append((p, dt))
            prev_dt = dt
        folder = f"{d:%Y-%m-%d}_S{series_idx:03d}_{start_dt:%H%M}-{prev_dt:%H%M}"
        groups.append((folder, list(cur)))
    return groups

def build_args():
    parser = argparse.ArgumentParser(description="Группировка JPG/MOV по сериям (время съёмки).")
    parser.add_argument("root", nargs="?", help="Корневая папка с файлами")
    parser.add_argument("--minutes", type=int, help=f"Порог разрыва, мин (по умолчанию {MINUTES})")
    parser.add_argument("--workers", type=int, help="Потоки чтения EXIF для JPG (по умолчанию авто)")
    parser.add_argument("--dry-run", action="store_true", help="Только план и лог, без перемещений")
    parser.add_argument("--mov-batch", type=int, help=f"Размер батча для exiftool по MOV (по умолчанию {MOV_BATCH})")
    if len(sys.argv) > 1 and any(a.strip() for a in sys.argv[1:]):
        args = parser.parse_args()
        args.root = Path(args.root).resolve() if args.root else Path(DEFAULT_ROOT).resolve()
        args.minutes = args.minutes if args.minutes is not None else MINUTES
        args.workers = args.workers if args.workers is not None else (_auto_workers() if WORKERS is None else WORKERS)
        args.dry_run = args.dry_run or DRY_RUN
        args.mov_batch = args.mov_batch if args.mov_batch is not None else MOV_BATCH
        return args
    args = argparse.Namespace()
    args.root = Path(DEFAULT_ROOT).resolve()
    args.minutes = MINUTES
    args.workers = (_auto_workers() if WORKERS is None else WORKERS)
    args.dry_run = DRY_RUN
    args.mov_batch = MOV_BATCH
    return args

def main():
    args = build_args()

    root: Path = args.root
    if not root.exists():
        print(f"Папка не найдена: {root}", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    all_media = list(walk_media(root))
    jpgs = [p for p in all_media if media_kind(p) == "jpg"]
    movs = [p for p in all_media if media_kind(p) == "mov"]
    print(f"Найдено всего: {len(all_media)} (JPG={len(jpgs)}, MOV={len(movs)}) под {root}")

    dated = []
    nodate = []

    # JPG - потоково
    if jpgs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(parse_exif_datetime_jpg, p): p for p in jpgs}
            done = 0
            for fut in as_completed(futs):
                p, dt, reason = fut.result()
                (dated if dt is not None else nodate).append((p, dt if dt else reason))
                done += 1
                if done % 5000 == 0:
                    print(f"JPG EXIF обработано: {done}/{len(jpgs)}")

    # MOV - батчами через exiftool
    if movs:
        exiftool_path = which_exiftool()
        if not exiftool_path:
            print("ВНИМАНИЕ: exiftool не найден — все MOV попадут в NoDate.", file=sys.stderr)
        d2, n2 = parse_mov_datetimes_exiftool(movs, exiftool_path, batch_size=args.mov_batch)
        dated.extend(d2)
        nodate.extend(n2)

    print(f"Дат успешно: {len(dated)}; без даты: {len(nodate)}")

    # Группировка и перенос
    groups = group_into_series(dated, args.minutes) if dated else []
    print(f"Серий сформировано: {len(groups)}")

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = root / f"series_moves_log_{ts}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["src_path", "dest_path", "exif_datetime", "note"])

        # Серии
        for folder_name, items in groups:
            dest_dir = root / folder_name
            if not args.dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for p, dt in items:
                dest_path = ""
                note = ""
                if not args.dry_run:
                    try:
                        dest_obj = ensure_unique_path(dest_dir, p.name)
                        shutil.move(str(p), str(dest_obj))
                        dest_path = str(dest_obj)
                    except Exception as e:
                        note = f"ERROR move: {e}"
                writer.writerow([str(p), dest_path, dt.strftime("%Y-%m-%d %H:%M:%S"), note])

        # NoDate
        if nodate:
            ndir = root / NODATE_DIRNAME
            if not args.dry_run:
                ndir.mkdir(parents=True, exist_ok=True)
            for item in nodate:
                p = item[0]
                reason = item[1] if isinstance(item[1], str) else ""
                dest_path = ""
                note = reason
                if not args.dry_run:
                    try:
                        dest_obj = ensure_unique_path(ndir, p.name)
                        shutil.move(str(p), str(dest_obj))
                        dest_path = str(dest_obj)
                    except Exception as e:
                        note = f"{reason}; ERROR move: {e}"
                writer.writerow([str(p), dest_path, "", note])

    t1 = time.time()
    print(f"Готово за {t1 - t0:.1f} сек. Лог: {log_path}")

if __name__ == "__main__":
    main()
