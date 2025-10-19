#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Файл gpoup_select_ai_check.py
Описание: Скрипт проверяет соответствие флага, закодированного в имени серии (часть после второго символа "_"),
и окончательной метки модели (поле CSV `ai_final_label`).
Для каждой несоответствующей серии выводится строка с деталями. В конце печатается сводная статистика.
Автор: naumenko33301 <naumenko33301@gmail.com>
Организация:
Дата создания: 2025-10-20
Версия: 1.0
Лицензия: All Rights Reserved
"""

from __future__ import annotations
import csv
import os
import sys
from typing import Optional, Set

# Параметры конфигурации
CSV_PATH = r"D:\xxxxxx\series_db.csv"  # путь к CSV-файлу с логом серий
PREFER_FIELD = "series_dirname"  # предпочитаемое поле для имени серии
ENCODINGS_TRY = ("utf-8-sig", "cp1251", "utf-8")  # список кодировок для попыток открытия файла

# Отображение флагов в допустимые классы модели
RU_FLAG_MAPPING = {
    "": {"empty"},           
    "марал": {"deer", "moose"},
    "маралсамец": {"deer", "moose"},
    "медведь": {"bear"},
    "человек": {"person"},
    "птица": {"bird"},
    "пусто": {"empty"},
    "пустой": {"empty"},
    "empty": {"empty"},
}


# Извлекает флаг из имени серии
def extract_flag_from_series_name(name: str) -> Optional[str]:
    if not name:
        return None
    base = os.path.basename(name.strip())
    idxs = [i for i, ch in enumerate(base) if ch == "_"]
    if len(idxs) < 2:
        return None
    sub = base[idxs[1] + 1 :]  # после второго "_"
    token = sub.split("_")[-1] if "_" in sub else sub
    return token.strip()


# Преобразует флаг в множество допустимых меток модели
def ru_flag_to_targets(flag_ru: str) -> Optional[Set[str]]:
    if flag_ru is None:
        return None
    ru = flag_ru.strip().lower()
    return RU_FLAG_MAPPING.get(ru)


def open_csv_any(path: str):
    last_err = None
    for enc in ENCODINGS_TRY:
        try:
            f = open(path, "r", encoding=enc, newline="")
            head = f.readline()
            if not head:
                raise ValueError("Пустой файл")
            f.seek(0)
            return f, enc
        except Exception as e:
            last_err = e
    raise last_err if last_err else FileNotFoundError(path)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"Файл не найден: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    total_rows = 0
    checked = 0
    correct = 0
    incorrect = 0
    unknown_flag = 0
    errors = []

    try:
        f, used_enc = open_csv_any(CSV_PATH)
    except Exception as e:
        print(f"Не удалось открыть CSV: {e}", file=sys.stderr)
        sys.exit(2)

    with f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"', doublequote=True, skipinitialspace=False)
        if "ai_final_label" not in reader.fieldnames:
            print("В CSV отсутствует обязательная колонка 'ai_final_label'", file=sys.stderr)
            sys.exit(3)

        for row in reader:
            total_rows += 1

            series_name = row.get(PREFER_FIELD) or row.get("series_dirname") or row.get("series_relpath")
            flag_ru = extract_flag_from_series_name(series_name or "")

            if flag_ru is None:
                continue

            targets = ru_flag_to_targets(flag_ru)
            if targets is None:
                unknown_flag += 1
                errors.append({
                    "series": series_name,
                    "flag_ru": flag_ru,
                    "targets": "UNKNOWN_FLAG",
                    "ai_final_label": (row.get("ai_final_label") or "").strip().lower(),
                })
                continue

            model_label = (row.get("ai_final_label") or "").strip().lower()
            if not model_label:
                model_label = (row.get("ai_top1_label") or "").strip().lower()

            checked += 1
            if model_label in targets:
                correct += 1
            else:
                incorrect += 1
                errors.append({
                    "series": series_name,
                    "flag_ru": flag_ru,
                    "targets": ",".join(sorted(targets)),
                    "ai_final_label": model_label or "<empty>",
                })

    for e in errors:
        print(f"[ERROR] {e['series']} | флаг='{e['flag_ru']}' -> допустимо: {{{e['targets']}}}; ai_final_label='{e['ai_final_label']}'")

    acc = (correct / checked * 100.0) if checked > 0 else 0.0
    print()
    print("Итоговая статистика:")
    print(f"Файл:                {CSV_PATH}")
    print(f"Кодировка:           {used_enc}")
    print(f"Всего строк:         {total_rows}")
    print(f"Проверено (с флагом):{checked}")
    print(f"Верно:               {correct}")
    print(f"С ошибками:          {incorrect}")
    if unknown_flag:
        print(f"Неизвестные флаги:   {unknown_flag} (не учитываются в метрике)")
    print(f"Точность:            {acc:.2f}%")

if __name__ == "__main__":
    main()
