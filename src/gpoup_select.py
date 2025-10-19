# -*- coding: utf-8 -*-
"""
Файл: gpoup_select.py
Описание: Валидация и анализ папок серий (JPG/MOV), сбор статистики и опциональный AI-анализ содержимого серий.
Автор: naumenko33301 <naumenko33301@gmail.com>
Дата создания: 2025-10-20
Версия: 1.0
Лицензия: All Rights Reserved
"""

"""
Проверка серий JPG/MOV и ИИ-анализ содержимого серий.
Все пути и флаги задаются в CONFIG (без CLI аргументов).

Ключевое:
- Валидация соответствия дат/временных интервалов именам папок.
- Сбор сводной статистики и запись в CSV-базу (опционально).
- ИИ-классификация (CLIP): фото + кадры из видео -> агрегирование по серии.

Функциональность:
1) Проверяет соответствие файлов интервалам папок серий (конец минуты включительно).
2) Собирает статистику по сериям и ведёт общую базу CSV.
3) ИИ-классификация: кадры из фото + семплы из .mov -> агрегация по серии.
4) Итоговый ответ по серии с учётом уверенности и частоты кадров.

Зависимости:
    pip install exifread
    pip install open-clip-torch pillow opencv-python numpy
    + PyTorch (CUDA)
    exiftool (для MOV дат; exiftool.exe в PATH или рядом со скриптом)
"""

import sys
try:
    import torch
    print(sys.executable)
    print(torch.__version__)
except Exception:
    print(sys.executable)
    print("torch not imported")

# Конфигурация
# Пути и отчёты
ROOT_DIR              = r"D:\Проект_Катя\6 солонец 2024"     # Корневой каталог с папками серий
REPORTS_DIR           = r"D:\Проект_Катя\отчёты"             # Куда класть summary/details; ""/None => в ROOT_DIR
SERIES_DB_PATH        = r"D:\Проект_Катя\6 солонец 2024.csv" # Общий CSV; ""/None => базу не вести
WRITE_DB_WITH_BOM     = True                                    # Писать базу в utf-8-sig (Excel-friendly)

# Поведение и производительность
CHECK_NODATE          = True                                     # Проверять папку NoDate
WORKERS               = None                                     # None=авто (4*CPU) для EXIF по JPG; или число
MOV_BATCH             = 800                                      # Размер батча для exiftool по .mov
PROGRESS_EVERY        = 100                                      # Прогресс по JPG EXIF

# Настройки ИИ
ENABLE_AI             = True                                     # Включить ИИ-определение объектов по сериям
MAX_IMAGES_PER_SERIES = 30                                       # Максимум фото на серию (равномерная выборка)
FRAMES_PER_VIDEO      = 4                                        # Сколько кадров извлекать из каждого .mov
CLIP_MODEL            = "ViT-L-14"                               # Модель open_clip
CLIP_PRETRAINED       = "openai"                                   # Датасет для предобучения модели ("openai", "laion2b_s32b_b82k" и др.)                                 
CLIP_BATCH            = 16                                       # Батч для инференса
SERIES_CONF_PRINT_TOPK= 3                                        # Сколько топ-классов печатать в консоль
FRAME_CONF_THRESHOLD  = 0.20                                     # Порог «кадр содержит класс» для счётчиков
AI_TOPK_TO_DB         = 3                                        # Сколько топ-классов писать в ai_top3
AI_ADD_COLUMNS        = True                                     # Добавлять новые колонки в базу (если False - только консоль)
MERGE_LABELS          = {"deer": ["elk"]}                        # Объединяем elk в deer (марал/олень в один класс)

# Агрегация по серии
# Итоговый score = W_PRESENCE * presence(noisy-OR) + W_SHARE * share_top1
W_PRESENCE            = 0.6                                       # Вес присутствия класса в серии (noisy-OR агрегация)
W_SHARE               = 0.4                                       # Вес доли кадров где класс топ-1
# Снижение «пересыщения» логитов кадра
LOGIT_TEMPERATURE     = 30.0                                      # Температура для нормализации логитов (чем выше - мягче распределение)                                     

# Подавление класса "empty"
EMPTY_KEEP_SHARE      = 0.50                                     # Если empty покрывает >=50% кадров - можно дать шанс
EMPTY_KEEP_PRESENCE   = 0.80                                     # Либо presence у empty >= 0.80 - можно дать шанс
EMPTY_KEEP_MARGIN     = 0.15                                     # Запас empty над лучшим не-empty по share/presence
NONEMPTY_STRONG_SHARE = 0.25                                     # Сильный сигнал не-empty по share
NONEMPTY_STRONG_PRESENCE= 0.50                                   # Сильный сигнал не-empty по presence
EMPTY_SUPPRESS_FACTOR = 0.30                                     # Во сколько раз ослабляем empty при подавлении
EMPTY_MIN_FACTOR      = 0.05                                     # Минимальный фактор подавления empty
PRINT_EMPTY_GATING_DEBUG = True                                  # Печатать причины подавления

# Few-shot / визуальные прототипы
ENABLE_FEWSHOT        = True                                     # Включить визуальные прототипы
CONTENT_DIR           = r"D:\Проект_Катя\content"                # Content/<folder>/<*.jpg|png|webp...>
# Маппинг: папка в content -> внутренняя метка модели
CONTENT_CLASS_MAP     = {
    "bear":  "bear",
    "empty": "empty",
    "person":"person",
    "maral": "deer",  # «марал» копим в целевой класс deer
}
CONTENT_MAX_IMAGES_PER_CLASS = 200                               # Ограничение на число эталонов на класс
PROTO_WEIGHT          = 0.55                                     # Вес прототипов при слиянии логитов (0..1)
PROTO_TEMPERATURE     = 30.0                                     # «Температура» для прототипных логитов

# Классы и текстовые промпты для zero-shot
AI_CLASSES = {
    "bear": [
        "a photo of a bear",
        "a brown bear in the forest",
        "a black bear in the wild"
    ],
    "deer": [
        "a photo of a deer",
        "a red deer in the forest",
        "a roe deer in the wild",
        "a white-tailed deer",
        "an elk (wapiti) in the forest",
        "a maral (Cervus canadensis)"
    ],
    "elk": [
        "an elk (wapiti)",
        "a maral with antlers"
    ],
    "moose": [
        "a moose (alces alces)"
    ],
    "boar": [
        "a wild boar",
        "a feral hog"
    ],
    "wolf": ["a wolf in the wild"],
    "fox": ["a fox in the forest"],
    "dog": ["a dog"],
    "cow": ["a cow"],
    "horse": ["a horse"],
        "person": [
        "a person with a visible human face (eyes, nose, mouth) partially occluded by branches",
        "a person with visible hands and five fingers",
        "a close-up of a human hand with fingers spread",
        "a close-up of a human face in a hood or mask (eyes and nose visible)",
        "a person in profile showing head, neck, and shoulders",
        "a back view of a person with head and shoulders visible",
        "a person walking upright on two legs",
        "a person running with arms swinging",
        "a person crouching or kneeling near a tree",
        "a person sitting on a log or ground",
        "lower body only: human legs with pants and boots",
        "upper body only: human torso with arms and hands visible",
        "a hiker with a backpack and trekking poles",
        "a person wearing a high-visibility vest or helmet",
        "a person wearing camouflage clothing (hands or face partially visible)",
        "a person holding a flashlight at night (beam visible)",
        "a person using a headlamp at night (glare toward the camera)",
        "a person holding a smartphone or camera in hand",
        "an infrared night-vision image of a person outdoors",
        "a thermal image of a person outdoors",
        "a human silhouette at night",
        "a motion-blurred person moving across the frame",
        "a person partially hidden behind an animal feeder or tree",
        "a person on a narrow trail with visible boots and calves",
        "a person with gloves (fingers shape still discernible)",
        "a person wearing a backpack viewed from behind",
        "a person in rain or snow with wet clothing and visible face or hands",
# -*- coding: utf-8 -*-
"""
Файл: gpoup_select.py
        "soft negative: a person present in a forest scene with no animals visible"
Описание: Валидация и анализ папок серий (JPG/MOV), сбор статистики и опциональный AI-анализ содержимого серий.
Автор: naumenko33301 <naumenko33301@gmail.com>mpty": ["This is not a deer, this is not a bear, this is a photo of a forest. This is not a deer, this is not a bear, this is a photo of a forest. This is not a deer, this is not a bear, this is a photo of a forest. This is not a deer, this is not a bear. Empty."]
Организация:
Дата создания: 2025-10-20
Версия: 1.0
Лицензия: All Rights Reservedimport csv
"""n
"""
Проверка серий JPG/MOV и ИИ-анализ содержимого серий.
Все пути и флаги задаются в CONFIG (без CLI аргументов).me

Ключевое:lib
- Валидация соответствия дат/временных интервалов именам папок.ics
- Сбор сводной статистики и запись в CSV-базу (опционально).
- ИИ-классификация (CLIP): фото + кадры из видео -> агрегирование по серии.
urrent.futures import ThreadPoolExecutor, as_completed
Функциональность:
1) Проверяет соответствие файлов интервалам папок серий (конец минуты включительно).
2) Собирает статистику по сериям и ведёт общую базу CSV.
3) ИИ-классификация: кадры из фото + семплы из .mov -> агрегация по серии.try:
4) Итоговый ответ по серии с учётом уверенности и частоты кадров.import exifread

Зависимости:библиотека 'exifread'. Установите: pip install exifread", file=sys.stderr)
    pip install exifread
    pip install open-clip-torch pillow opencv-python numpy
    + PyTorch (CUDA)# ИИ зависимости - грузим лениво, чтобы скрипт работал и без них (если ENABLE_AI=False)
    exiftool (для MOV дат; exiftool.exe в PATH или рядом со скриптом)
"""

import sys
try: = None
    import torch
    print(sys.executable)
    print(torch.__version__)def _lazy_import_ai():
except Exception: _torch, _cv2, _np, _PIL_Image, _PIL
    print(sys.executable)
    print("torch not imported")s _open_clip  # type: ignore

# Конфигурацияs _torch  # type: ignore
# Пути и отчёты
ROOT_DIR              = r"D:\Проект_Катя\6 солонец 2024"     # Корневой каталог с папками серийs _cv2  # type: ignore
REPORTS_DIR           = r"D:\Проект_Катя\отчёты"             # Куда класть summary/details; ""/None => в ROOT_DIR
SERIES_DB_PATH        = r"D:\Проект_Катя\6 солонец 2024.csv" # Общий CSV; ""/None => базу не вестиy as _np  # type: ignore
WRITE_DB_WITH_BOM     = True                                    # Писать базу в utf-8-sig (Excel-friendly)
ort Image as _PIL_Image  # type: ignore
# Поведение и производительность
CHECK_NODATE          = True                                     # Проверять папку NoDate
WORKERS               = None                                     # None=авто (4*CPU) для EXIF по JPG; или число# Папка серии
MOV_BATCH             = 800                                      # Размер батча для exiftool по .mov = re.compile(
PROGRESS_EVERY        = 100                                      # Прогресс по JPG EXIF-\d{2})_S(?P<idx>\d{3})_(?P<t1>\d{4})-(?P<t2>\d{4})(?:_(?P<tags>.*))?$"

# Настройки ИИODATE_DIRNAME = "NoDate"
ENABLE_AI             = True                                     # Включить ИИ-определение объектов по сериям
MAX_IMAGES_PER_SERIES = 30                                       # Максимум фото на серию (равномерная выборка)EXT_JPG = {".jpg", ".jpeg", ".jpe", ".jfif", ".jif", ".pjpeg", ".pjp"}
FRAMES_PER_VIDEO      = 4                                        # Сколько кадров извлекать из каждого .mov
CLIP_MODEL            = "ViT-L-14"                               # Модель open_clip
CLIP_PRETRAINED       = "openai"                                   # Датасет для предобучения модели ("openai", "laion2b_s32b_b82k" и др.)                                 def _auto_workers():
CLIP_BATCH            = 16                                       # Батч для инференсаs.cpu_count() or 8) * 4)
SERIES_CONF_PRINT_TOPK= 3                                        # Сколько топ-классов печатать в консоль
FRAME_CONF_THRESHOLD  = 0.20                                     # Порог «кадр содержит класс» для счётчиковdef media_kind(path: Path) -> str:
AI_TOPK_TO_DB         = 3                                        # Сколько топ-классов писать в ai_top3
MERGE_LABELS          = {"deer": ["elk"]}                        # Объединяем elk в deer (марал/олень в один класс)                              # Добавлять новые колонки в базу (если False - только консоль) "jpg"

# Агрегация по серии
# Итоговый score = W_PRESENCE * presence(noisy-OR) + W_SHARE * share_top1
W_PRESENCE            = 0.6                                       # Вес присутствия класса в серии (noisy-OR агрегация)
W_SHARE               = 0.4                                       # Вес доли кадров где класс топ-1
# Снижение «пересыщения» логитов кадра
LOGIT_TEMPERATURE     = 30.0                                      # Температура для нормализации логитов (чем выше - мягче распределение)                                     

# Подавление класса "empty"
EMPTY_KEEP_SHARE      = 0.50                                     # Если empty покрывает >=50% кадров - можно дать шанс
EMPTY_KEEP_PRESENCE   = 0.80                                     # Либо presence у empty >= 0.80 - можно дать шанс
EMPTY_KEEP_MARGIN     = 0.15                                     # Запас empty над лучшим не-empty по share/presence
NONEMPTY_STRONG_SHARE = 0.25                                     # Сильный сигнал не-empty по share
NONEMPTY_STRONG_PRESENCE= 0.50                                   # Сильный сигнал не-empty по presence
EMPTY_SUPPRESS_FACTOR = 0.30                                     # Во сколько раз ослабляем empty при подавлении
EMPTY_MIN_FACTOR      = 0.05                                     # Минимальный фактор подавления empty
PRINT_EMPTY_GATING_DEBUG = True                                  # Печатать причины подавления

# Few-shot / визуальные прототипы
ENABLE_FEWSHOT        = True                                     # Включить визуальные прототипы
CONTENT_DIR           = r"D:\Проект_Катя\content"                # Content/<folder>/<*.jpg|png|webp...>
# Маппинг: папка в content -> внутренняя метка модели
CONTENT_CLASS_MAP     = {
    "bear":  "bear",
    "empty": "empty",
    "person":"person",
    "maral": "deer",  # «марал» копим в целевой класс deer
}
CONTENT_MAX_IMAGES_PER_CLASS = 200                               # Ограничение на число эталонов на класс
PROTO_WEIGHT          = 0.55                                     # Вес прототипов при слиянии логитов (0..1)
PROTO_TEMPERATURE     = 30.0                                     # «Температура» для прототипных логитов

# Классы и текстовые промпты для zero-shot
AI_CLASSES = {
    "bear": [
        "a photo of a bear",
        "a brown bear in the forest",
        "a black bear in the wild"
    ],
    "deer": [
        "a photo of a deer",
        "a red deer in the forest",
        "a roe deer in the wild",
        "a white-tailed deer",
        "an elk (wapiti) in the forest",
        "a maral (Cervus canadensis)"
    ],
    "elk": [
        "an elk (wapiti)",
        "a maral with antlers"
    ],
    "moose": [
        "a moose (alces alces)"
    ],
    "boar": [
        "a wild boar",
        "a feral hog"
    ],
    "wolf": ["a wolf in the wild"],
    "fox": ["a fox in the forest"],
    "dog": ["a dog"],
    "cow": ["a cow"],
    "horse": ["a horse"],
        "person": [
        "a person with a visible human face (eyes, nose, mouth) partially occluded by branches",
        "a person with visible hands and five fingers",
        "a close-up of a human hand with fingers spread",
        "a close-up of a human face in a hood or mask (eyes and nose visible)",
        "a person in profile showing head, neck, and shoulders",
        "a back view of a person with head and shoulders visible",
        "a person walking upright on two legs",
        "a person running with arms swinging",
        "a person crouching or kneeling near a tree",
        "a person sitting on a log or ground",
        "lower body only: human legs with pants and boots",
        "upper body only: human torso with arms and hands visible",
        "a hiker with a backpack and trekking poles",
        "a person wearing a high-visibility vest or helmet",
        "a person wearing camouflage clothing (hands or face partially visible)",
        "a person holding a flashlight at night (beam visible)",
        "a person using a headlamp at night (glare toward the camera)",
        "a person holding a smartphone or camera in hand",
        "an infrared night-vision image of a person outdoors",
        "a thermal image of a person outdoors",
        "a human silhouette at night",
        "a motion-blurred person moving across the frame",
        "a person partially hidden behind an animal feeder or tree",
        "a person on a narrow trail with visible boots and calves",
        "a person with gloves (fingers shape still discernible)",
        "a person wearing a backpack viewed from behind",
        "a person in rain or snow with wet clothing and visible face or hands",
                data = json.loads(res.stdout)
            except Exception as e:
                for p in chunk:
                    nodate[p] = f"exiftool JSON parse error: {e}"
                continue
            by_src = { Path(d.get("SourceFile")).resolve(): d
                       for d in data if isinstance(d, dict) and d.get("SourceFile") }
            for p in chunk:
                rec = by_src.get(p.resolve())
                if not rec:
                    nodate[p] = "no record from exiftool"
                    continue
                if "Error" in rec and rec["Error"]:
                    nodate[p] = f"exiftool: {rec['Error']}"
                    continue
                dt_str = rec.get("DateTimeOriginal") or rec.get("CreateDate") or rec.get("MediaCreateDate")
                if not dt_str:
                    nodate[p] = "No CreateDate/MediaCreateDate/DateTimeOriginal"
                    continue
                dt = parse_dt_string(dt_str)
                if not dt:
                    nodate[p] = f"Bad date format: {dt_str!r}"
                else:
                    dated[p] = dt
        except Exception as e:
            for p in chunk:
                nodate[p] = f"exiftool error: {e}"
    return dated, nodate

# Утилиты
def collect_targets(root: Path):
    series_dirs = []
    nodate_dir = None
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir(follow_symlinks=False):
                continue
            name = entry.name
            if name == NODATE_DIRNAME:
                nodate_dir = Path(entry.path)
                continue
            if SERIES_DIR_RE.match(name):
                series_dirs.append(Path(entry.path))
    return series_dirs, nodate_dir

def gaps_seconds(sorted_datetimes):
    if len(sorted_datetimes) < 2:
        return [], None, None, None, None
    diffs = []
    prev = sorted_datetimes[0]
    for cur in sorted_datetimes[1:]:
        diffs.append((cur - prev).total_seconds())
        prev = cur
    mn = min(diffs); mx = max(diffs)
    mean = sum(diffs) / len(diffs)
    med = statistics.median(diffs)
    return diffs, mn, mx, mean, med

def weekday_name(day):
    names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return names[day.isoweekday()-1]

# Вычисляет энтропию распределения вероятностей
def calculate_ai_entropy(probs_dict):
    import math
    entropy = 0.0
    for prob in probs_dict.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Вычисляет долю кадров, где топ-1 класс совпадает с финальным
def calculate_ai_consensus(per_frame, final_label):
    if not per_frame:
        return 0.0
    consensus_count = 0
    for frame_dist in per_frame:
        top_class = max(frame_dist.items(), key=lambda kv: kv[1])[0]
        if top_class == final_label:
            consensus_count += 1
    return consensus_count / len(per_frame)

# Возвращает категорию времени дня
def get_time_of_day_category(hour):
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

# Определяет интенсивность активности
def calculate_activity_intensity(files_per_hour):
    if files_per_hour < 10:
        return "low"
    elif files_per_hour < 30:
        return "medium"
    else:
        return "high"

# Определяет наличие всплесков активности (короткие интервалы)
def detect_burst_activity(gaps_seconds, threshold_sec=60):
    if not gaps_seconds:
        return 0
    short_gaps = sum(1 for gap in gaps_seconds if gap < threshold_sec)
    return 1 if short_gaps > len(gaps_seconds) * 0.3 else 0

# Определяет наличие аномальных интервалов
def detect_gap_outliers(gaps_seconds):
    if len(gaps_seconds) < 3:
        return 0
    import statistics
    median_gap = statistics.median(gaps_seconds)
    outliers = sum(1 for gap in gaps_seconds if gap > median_gap * 3 or gap < median_gap * 0.1)
    return 1 if outliers > 0 else 0

# Вычисляет надежность предсказания (0-1)
def calculate_prediction_reliability(consensus, confidence_gap, entropy, samples):
    if samples == 0:
        return 0.0
    
    # Базовые компоненты
    consensus_score = consensus  # 0-1
    confidence_score = min(confidence_gap * 2, 1.0)  # нормализуем к 0-1
    entropy_score = max(0, 1.0 - entropy / 5.0)  # энтропия 0-5, инвертируем
    
    # Веса компонентов
    reliability = (0.4 * consensus_score + 0.4 * confidence_score + 0.2 * entropy_score)
    return min(max(reliability, 0.0), 1.0)

# Определяет качество детекции (high/medium/low)
def calculate_detection_quality(consensus, entropy, samples, final_prob):
    if samples == 0:
        return "unknown"
    
    # Комбинированная оценка
    quality_score = (consensus * 0.5 + (1.0 - entropy/5.0) * 0.3 + final_prob * 0.2)
    
    if quality_score >= 0.7:
        return "high"
    elif quality_score >= 0.4:
        return "medium"
    else:
        return "low"

# Подсчитывает количество классов с вероятностью выше порога
def calculate_ai_class_diversity(probs_dict, threshold=0.1):
    return sum(1 for prob in probs_dict.values() if prob >= threshold)

# Определяет уровень неопределенности по энтропии
def calculate_ai_uncertainty_level(entropy):
    if entropy < 2.0:
        return "low"
    elif entropy < 3.5:
        return "medium"
    else:
        return "high"

def write_series_db_rows(db_path: Path, rows, with_bom=False):
    need_header = not db_path.exists() or db_path.stat().st_size == 0
    enc = "utf-8-sig" if with_bom else "utf-8"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "run_ts", "root_path",
        "series_relpath", "series_dirname", "series_uid",
        "date", "weekday", "start_hhmm", "end_hhmm",
        "first_dt", "last_dt", "duration_sec",
        "files_total", "jpg_count", "mov_count",
        "errors_in_series",
        "min_gap_sec", "max_gap_sec", "mean_gap_sec", "median_gap_sec",
        "hour_of_day", "is_weekend", "time_of_day_category",
        "session_length_hours", "files_per_minute", "files_per_hour",
        "jpg_mov_ratio", "error_rate", "date_consistency",
        "gap_outliers", "activity_intensity", "burst_activity"
    ]
    ai_cols = [
        "ai_model","ai_samples",
        "ai_top1_label","ai_top1_series_prob","ai_top3",
        "ai_counts_json",
        "ai_final_label","ai_final_prob",
        "ai_consensus", "ai_entropy", "ai_confidence_gap", "ai_second_place_gap",
        "prediction_reliability", "detection_quality", "ai_class_diversity", "ai_uncertainty_level"
    ]
    cols = base_cols + (ai_cols if AI_ADD_COLUMNS else [])
    with open(db_path, "a", newline="", encoding=enc) as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

# Равномерная выборка k элементов по индексу (если k>=len -> все)
def even_sample(items, k):
    n = len(items)
    if k >= n:
        return list(items)
    out = []
    for i in range(k):
        idx = math.floor(i * n / k)
        out.append(items[idx])
    if out and out[-1] != items[-1]:
        out[-1] = items[-1]
    return out

# AI: CLIP zero-shot
class SeriesAI:
    def __init__(self, model_name, pretrained, classes_dict, device=None, batch=16):
        _lazy_import_ai()
        self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = _open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = _open_clip.get_tokenizer(model_name)
        self.model.eval()
        # Build class embeddings (среднее по промптам для каждого класса)
        with _torch.no_grad():
            class_embs = []
            self.class_names = []
            for cls, prompts in classes_dict.items():
                toks = self.tokenizer(prompts)
                toks = toks.to(self.device)
                txt = self.model.encode_text(toks)
                txt = txt / txt.norm(dim=-1, keepdim=True)
                cls_emb = txt.mean(dim=0, keepdim=True)
                cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
                class_embs.append(cls_emb)
                self.class_names.append(cls)
            self.class_matrix = _torch.cat(class_embs, dim=0)  # [C, D]
        self.batch = batch

        # Визуальные прототипы из content
        self.proto_matrix = None   # [C,D] выровнено по self.class_names
        self.proto_mask = None     # [C] 1 там, где есть прототип, иначе 0
        if ENABLE_FEWSHOT and CONTENT_DIR:
            try:
                self._init_visual_prototypes(
                    CONTENT_DIR, CONTENT_CLASS_MAP, CONTENT_MAX_IMAGES_PER_CLASS
                )
            except Exception as e:
                print(f"[AI] Прототипы не подключены: {e}", file=sys.stderr)

    # Собирает по папкам mean-эмбеддинги изображений и кладёт в self.proto_matrix
    def _init_visual_prototypes(self, content_dir, folder2label, max_per_class):
        _lazy_import_ai()
        from pathlib import Path
        from PIL import Image

        root = Path(content_dir)
        if not root.exists():
            print(f"[AI] CONTENT_DIR не найден: {root}", file=sys.stderr)
            return

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        # Копим эмбеддинги эталонов по ЦЕЛЕВЫМ меткам модели
        by_label = {name: [] for name in self.class_names}

        def _files_in(d: Path):
            return [p for p in d.rglob("*") if p.suffix.lower() in exts and p.is_file()]

        # Обходим папки в content и считаем средние L2-нормированные эмбеддинги
        for folder, target_label in folder2label.items():
            if target_label not in by_label:
                print(f"[AI] Пропускаю content/{folder}: не знаю метку '{target_label}'", file=sys.stderr)
                continue
            d = root / folder
            if not d.exists():
                print(f"[AI] Пропускаю content/{folder}: нет папки", file=sys.stderr)
                continue
            files = _files_in(d)
            if not files:
                print(f"[AI] Пропускаю content/{folder}: нет картинок", file=sys.stderr)
                continue
            # Равномерно сэмплируем, чтобы не было перекоса по сериям
            sel = even_sample(files, max_per_class)

            feats = []
            batch_pil = []
            for p in sel:
                try:
                    batch_pil.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
                if len(batch_pil) == self.batch:
                    f = self._encode_images(batch_pil)   # Уже L2-нормированы
                    feats.append(f); batch_pil = []
            if batch_pil:
                feats.append(self._encode_images(batch_pil))
            if not feats:
                continue
            mat = _torch.cat(feats, dim=0)            # [N,D]
            proto = mat.mean(dim=0, keepdim=True)     # [1,D]
            proto = proto / proto.norm(dim=-1, keepdim=True)
            by_label[target_label].append(proto)

        # Формируем матрицу [C,D] в порядке self.class_names
        C = len(self.class_names)
        D = self.class_matrix.shape[1]
        proto_mat = _torch.zeros((C, D), device=self.device)
        mask = _torch.zeros((C,), device=self.device)
        filled = 0
        for j, name in enumerate(self.class_names):
            vecs = by_label.get(name) or []
            if vecs:
                m = _torch.cat(vecs, dim=0).mean(dim=0, keepdim=True)
                m = m / m.norm(dim=-1, keepdim=True)
                proto_mat[j:j+1, :] = m
                mask[j] = 1.0
                filled += 1

        if filled == 0:
            print("[AI] В content нет валидных прототипов — работаем как обычно")
            return

        self.proto_matrix = proto_mat
        self.proto_mask = mask  # 1 - есть прототип, 0 - нет
        print(f"[AI] Прототипы загружены: {filled}/{C} классов из {root}")

    @staticmethod
    def _pil_from_bgr(bgr):
        _lazy_import_ai()
        rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
        return _PIL_Image.fromarray(rgb)

    def _encode_images(self, pil_images):
        _lazy_import_ai()
        with _torch.no_grad():
            tens = _torch.stack([self.preprocess(im.convert("RGB")) for im in pil_images]).to(self.device)
            img_feat = self.model.encode_image(tens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [N,D]
            return img_feat

    # Возвращает список дистрибуций по классам (list of dict). Ансамбль: текст + визуальные прототипы
    def infer_images(self, pil_images):
        _lazy_import_ai()
        outs = []
        with _torch.no_grad():
            for i in range(0, len(pil_images), self.batch):
                chunk = pil_images[i:i + self.batch]
                if not chunk:
                    break
                img_feat = self._encode_images(chunk)  # [N,D], L2-нормировано

                # Текстовые логиты
                logits_text = (img_feat @ self.class_matrix.T) * LOGIT_TEMPERATURE  # [N,C]
                logits = logits_text

                # Прототипы
                if getattr(self, "proto_matrix", None) is not None and getattr(self, "proto_mask", None) is not None:
                    proto_logits_raw = (img_feat @ self.proto_matrix.T)  # [N,C]
                    mask = self.proto_mask  # [C]
                    denom = mask.sum()
                    if denom > 0:
                        # Центрируем по доступным классам -> «0» становится нейтральным добавлением
                        mean_vals = (proto_logits_raw * mask).sum(dim=-1, keepdim=True) / denom
                        proto_logits_centered = (proto_logits_raw - mean_vals) * mask
                        logits = (1.0 - PROTO_WEIGHT) * logits_text + PROTO_WEIGHT * (
                                    PROTO_TEMPERATURE * proto_logits_centered)

                probs = logits.softmax(dim=-1).cpu().numpy()
                for row in probs:
                    outs.append({self.class_names[j]: float(row[j]) for j in range(len(self.class_names))})
        return outs


    def aggregate_series(self, per_frame):
        """
        Возвращает три агрегата:
        - presence (noisy-OR по кадрам)
        - mean_prob (средняя вероятность по кадрам)
        - share_top1 (доля кадров, где класс топ-1 при пороге)
        + counts по кадрам.
        """
        C = len(self.class_names)
        names = self.class_names
        # Presence (noisy-OR)
        prod = [1.0]*C
        for dist in per_frame:
            for j, name in enumerate(names):
                prod[j] *= (1.0 - dist[name])
        presence = {names[j]: 1.0 - prod[j] for j in range(C)}

        # Mean prob
        if per_frame:
            mean_prob = {name: float(sum(d[name] for d in per_frame))/len(per_frame) for name in names}
        else:
            mean_prob = {name: 0.0 for name in names}

        # Counts / share_top1
        counts = {name: 0 for name in names}
        for dist in per_frame:
            best = max(dist.items(), key=lambda kv: kv[1])
            if best[1] >= FRAME_CONF_THRESHOLD:
                counts[best[0]] += 1
        share_top1 = {name: (counts[name]/len(per_frame) if per_frame else 0.0) for name in names}

        return presence, mean_prob, share_top1, counts

    # Полный пайплайн: пер-кадр -> агрегаты
    def infer_series(self, images_pil):
        per_frame = self.infer_images(images_pil) if images_pil else []
        presence, mean_prob, share_top1, counts = self.aggregate_series(per_frame)
        top_presence = sorted(presence.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "per_frame": per_frame,
            "presence": presence,
            "mean": mean_prob,
            "share": share_top1,
            "counts": counts,
            "top": top_presence,          
            "series_probs": presence,    
            "samples": len(per_frame)
        }

# Логика объединения меток (elk -> deer)
def merge_labels(res, merge_map):
    probs = dict(res["series_probs"])
    counts = dict(res["counts"])
    presence = dict(res.get("presence", {}))
    meanp = dict(res.get("mean", {}))
    share = dict(res.get("share", {}))

    for tgt, sources in merge_map.items():
        base_not = 1.0 - presence.get(tgt, probs.get(tgt, 0.0))
        for s in sources:
            base_not *= (1.0 - presence.pop(s, probs.pop(s, 0.0)))
        presence[tgt] = 1.0 - base_not

        for s in sources:
            meanp[tgt] = meanp.get(tgt, 0.0) + meanp.pop(s, 0.0)
            share[tgt] = share.get(tgt, 0.0) + share.pop(s, 0.0)
            counts[tgt] = counts.get(tgt, 0) + counts.pop(s, 0)

    # Нормируем mean/ share после слияния
    def _sorted_top(d):
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "per_frame": res.get("per_frame", []),
        "presence": presence,
        "mean": meanp,
        "share": share,
        "counts": counts,
        "series_probs": presence,  
        "top": _sorted_top(presence),
        "samples": res.get("samples", 0)
    }

# Возвращает (suppress_empty: bool, factor: float, reason: str)
# factor применяется к вкладy empty в финальном скоринге
def apply_empty_gating(presence, share):
    empty_share = share.get("empty", 0.0)
    empty_presence = presence.get("empty", 0.0)
    nonempty_keys = [k for k in presence.keys() if k != "empty"]

    max_nonempty_share = max((share.get(k, 0.0) for k in nonempty_keys), default=0.0)
    max_nonempty_presence = max((presence.get(k, 0.0) for k in nonempty_keys), default=0.0)

    # Если есть любой не-empty сигнал - подавляем empty более агрессивно
    if max_nonempty_share > 0.05 or max_nonempty_presence > 0.05:
        # Если есть слабый сигнал не-empty, но empty всё ещё лидирует
        if empty_share > max_nonempty_share or empty_presence > max_nonempty_presence:
            return True, max(EMPTY_MIN_FACTOR, EMPTY_SUPPRESS_FACTOR * 0.7), "weak_nonempty_suppress"

    # Явное доминирование empty - не подавляем только при очень высоких показателях
    empty_clearly_dominates = (
        (empty_share >= max_nonempty_share + EMPTY_KEEP_MARGIN) or
        (empty_presence >= max_nonempty_presence + EMPTY_KEEP_MARGIN)
    )
    empty_big_enough = (empty_share >= EMPTY_KEEP_SHARE) or (empty_presence >= EMPTY_KEEP_PRESENCE)

    if empty_clearly_dominates and empty_big_enough:
        return False, 1.0, "empty_clearly_dominates"

    # Сильный сигнал не-empty - подавляем
    strong_nonempty = (max_nonempty_share >= NONEMPTY_STRONG_SHARE) or (max_nonempty_presence >= NONEMPTY_STRONG_PRESENCE)
    if strong_nonempty:
        return True, max(EMPTY_MIN_FACTOR, EMPTY_SUPPRESS_FACTOR), "strong_nonempty_signal"

    # По умолчанию подавляем empty, если он не дотягивает до keep-порогов
    if not empty_big_enough:
        return True, 0.4, "weak_empty_suppress"  
    return False, 1.0, "no_suppression"

# Точка входа
def main():
    root = Path(ROOT_DIR).resolve()
    if not root.exists():
        print(f"Папка не найдена: {root}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(REPORTS_DIR).resolve() if REPORTS_DIR else root
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    series_dirs, nodate_dir = collect_targets(root)
    if not series_dirs and not (CHECK_NODATE and nodate_dir and nodate_dir.exists()):
        print("Не найдено папок серий и NoDate. Проверьте корень.", file=sys.stderr)
        sys.exit(1)

    # Собираем файлы
    all_paths = []
    per_dir_files = {}
    for d in series_dirs:
        parsed = parse_series_interval(d.name)
        if not parsed:
            per_dir_files[d] = []
            continue
        files = []
        with os.scandir(d) as it:
            for e in it:
                if e.is_file(follow_symlinks=False):
                    p = Path(e.path)
                    if media_kind(p) in ("jpg", "mov"):
                        files.append(p)
        per_dir_files[d] = files
        all_paths.extend(files)

    nodate_files = []
    if CHECK_NODATE and nodate_dir and nodate_dir.exists():
        with os.scandir(nodate_dir) as it:
            for e in it:
                if e.is_file(follow_symlinks=False):
                    p = Path(e.path)
                    if media_kind(p) in ("jpg", "mov"):
                        nodate_files.append(p)
        all_paths.extend(nodate_files)

    print(f"К проверке файлов: {len(all_paths)} (в сериях: {sum(len(v) for v in per_dir_files.values())}, в NoDate: {len(nodate_files)})")

    # EXIF даты
    jpgs = [p for p in all_paths if media_kind(p) == "jpg"]
    movs = [p for p in all_paths if media_kind(p) == "mov"]

    workers = (_auto_workers() if WORKERS is None else WORKERS)

    meta_dt = {}
    meta_reason = {}

    # JPG
    if jpgs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(parse_exif_datetime_jpg, p): p for p in jpgs}
            done = 0
            for fut in as_completed(futs):
                p, dt, reason = fut.result()
                if dt is not None:
                    meta_dt[p] = dt
                else:
                    meta_reason[p] = reason
                done += 1
                if done % PROGRESS_EVERY == 0 or done == len(jpgs):
                    print(f"JPG EXIF прочитано: {done}/{len(jpgs)}")

    # MOV
    if movs:
        exiftool_path = which_exiftool()
        if not exiftool_path:
            print("ВНИМАНИЕ: exiftool не найден — MOV не будут валидно проверены на дату.", file=sys.stderr)
        dmap, nmap = exiftool_batch_mov(movs, exiftool_path, batch_size=MOV_BATCH)
        meta_dt.update(dmap)
        meta_reason.update(nmap)

    # Проверка + базовая статистика
    summary_rows = []
    details_rows = []
    db_rows = []
    total_errors = 0
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ИИ модель (лениво)
    print("[AI] Подготовка модели… первый запуск может скачать ~1.7 ГБ (ViT-L/14).")
    ai = None
    if ENABLE_AI:
        try:
            _lazy_import_ai()
            ai = SeriesAI(CLIP_MODEL, CLIP_PRETRAINED, AI_CLASSES, batch=CLIP_BATCH)
            print(f"[AI] Loaded {CLIP_MODEL} ({CLIP_PRETRAINED}) on {ai.device}")
        except Exception as e:
            print(f"[AI] Не удалось загрузить модель: {e}. Продолжаю без AI.", file=sys.stderr)
            ai = None

    # Хелпер выбор кадров для AI
    def collect_series_samples(series_paths):
        if not ENABLE_AI or ai is None:
            return []
        _lazy_import_ai()
        dated = [(p, meta_dt.get(p)) for p in series_paths if p in meta_dt]
        others = [p for p in series_paths if p not in meta_dt]
        dated.sort(key=lambda x: x[1])
        ordered = [p for p,_ in dated] + others
        imgs = [p for p in ordered if media_kind(p) == "jpg"]
        imgs = even_sample(imgs, MAX_IMAGES_PER_SERIES)
        frames = []
        for p in imgs:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                from PIL import Image
                img = Image.open(io.BytesIO(data)).convert("RGB")
                frames.append(img)
            except Exception:
                continue
        videos = [p for p in ordered if media_kind(p) == "mov"]
        for vp in videos:
            try:
                cap = _cv2.VideoCapture(str(vp))
                if not cap.isOpened():
                    continue
                frames_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                if frames_count <= 0:
                    step = max(1, int(cap.get(_cv2.CAP_PROP_FPS) or 10))
                    wanted = [i*step for i in range(FRAMES_PER_VIDEO)]
                else:
                    wanted = even_sample(list(range(frames_count)), FRAMES_PER_VIDEO)
                for idx in wanted:
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    pil = SeriesAI._pil_from_bgr(frame)
                    frames.append(pil)
                cap.release()
            except Exception:
                continue
        return frames

    # Основной цикл по сериям
    for d, files in per_dir_files.items():
        parsed = parse_series_interval(d.name)
        if not parsed:
            details_rows.append([str(d), "", "", "", "DIR_NAME_INVALID", "Имя папки не соответствует шаблону"])
            total_errors += 1
            continue
        day, start_dt, end_dt, start_hhmm, end_hhmm, idx = parsed

        err_count = 0
        times = []
        jpg_count = 0
        mov_count = 0

        for p in files:
            kind = media_kind(p)
            if kind == "jpg": jpg_count += 1
            elif kind == "mov": mov_count += 1

            dt = meta_dt.get(p)
            if not dt:
                reason = meta_reason.get(p, "No date")
                details_rows.append([str(d), p.name, "", f"{start_hhmm[:2]}:{start_hhmm[2:]}-{end_hhmm[:2]}:{end_hhmm[2:]}", "NO_DATE", reason])
                err_count += 1
                continue
            times.append(dt)
            if dt.date() != day:
                details_rows.append([str(d), p.name, dt.strftime("%Y-%m-%d %H:%M:%S"),
                                     f"{start_dt:%Y-%m-%d %H:%M}-{end_dt:%H:%M}",
                                     "DAY_MISMATCH", "Дата файла не совпадает с днём папки"])
                err_count += 1
            elif not (start_dt <= dt <= end_dt):
                details_rows.append([str(d), p.name, dt.strftime("%Y-%m-%d %H:%M:%S"),
                                     f"{start_hhmm[:2]}:{start_hhmm[2:]}-{end_hhmm[:2]}:{end_hhmm[2:]}",
                                     "OUT_OF_INTERVAL", "Время вне интервала папки"])
                err_count += 1

        files_total = len(files)
        if times:
            times.sort()
            mn = times[0]; mx = times[-1]
            duration_sec = (mx - mn).total_seconds()
            diffs, gmin, gmax, gmean, gmed = gaps_seconds(times)
            summary_rows.append([str(d), files_total, err_count, mn.strftime("%H:%M:%S"), mx.strftime("%H:%M:%S")])
        else:
            mn = mx = None
            duration_sec = ""
            gmin = gmax = gmean = gmed = ""
            summary_rows.append([str(d), files_total, err_count, "", ""])
        total_errors += err_count

        # AI по серии
        ai_top1_label = ""
        ai_top1_prob = ""
        ai_topk_str = ""
        ai_counts_json = ""
        ai_samples = 0
        ai_model_name = ""
        ai_final_label = ""
        ai_final_prob = ""

        if ENABLE_AI and ai is not None and files_total > 0:
            imgs_pil = collect_series_samples(files)
            res = ai.infer_series(imgs_pil)
            if MERGE_LABELS:
                res = merge_labels(res, MERGE_LABELS)

            # Series aggregates
            presence = res["presence"]
            share = res["share"]
            counts = res["counts"]
            ai_samples = res["samples"]

            # Базовый top-3 (по presence) - для совместимости/наглядности
            top_presence = sorted(presence.items(), key=lambda kv: kv[1], reverse=True)
            ai_model_name = f"{CLIP_MODEL}_{CLIP_PRETRAINED}"
            if ai_samples > 0:
                ai_top1_label = top_presence[0][0]
                ai_top1_prob = f"{top_presence[0][1]:.4f}"
                ai_topk_str = ";".join([f"{n}:{p:.4f}" for n, p in top_presence[:AI_TOPK_TO_DB]])
                ai_counts_json = json.dumps(counts, ensure_ascii=False)


            # ИТОГОВЫЙ ОТВЕТ (учёт частоты кадров)
            # Score = W_PRESENCE * presence + W_SHARE * share_top1
            # Исключим empty из соревнования, если есть другие классы
            candidate_names = list(presence.keys())  

            # Базовые баллы
            scores = {k: (W_PRESENCE * presence.get(k, 0.0) + W_SHARE * share.get(k, 0.0))
                      for k in candidate_names}

            # Подавление empty по правилам
            suppress_empty, empty_factor, empty_reason = False, 1.0, "n/a"
            if "empty" in scores:
                suppress_empty, empty_factor, empty_reason = apply_empty_gating(presence, share)
                if suppress_empty:
                    scores["empty"] *= empty_factor
                    if PRINT_EMPTY_GATING_DEBUG:
                        print(f"[EMPTY_GATING] {d.name}: {empty_reason} (factor={empty_factor:.2f})")

            # Нормализация и выбор финала
            ssum = sum(scores.values()) or 1.0
            norm_scores = {k: v / ssum for k, v in scores.items()}
            final_sorted = sorted(norm_scores.items(), key=lambda kv: kv[1], reverse=True)

            if final_sorted:
                ai_final_label = final_sorted[0][0]
                ai_final_prob = f"{final_sorted[0][1]:.4f}"


            top_show = "; ".join([f"{n}:{presence[n]:.2f}" for n,_ in top_presence[:SERIES_CONF_PRINT_TOPK]])
            counts_show = ", ".join([f"{n}={counts[n]}" for n in ["deer","moose","bear","person","empty"] if n in counts])

        # Временные метрики
        hour_of_day = int(start_hhmm[:2]) if start_hhmm else ""
        is_weekend = 1 if weekday_name(day) in ["Sat", "Sun"] else 0
        time_of_day_category = get_time_of_day_category(hour_of_day) if hour_of_day != "" else ""
        
        # Метрики активности
        session_length_hours = (mx - mn).total_seconds() / 3600 if times else 0
        files_per_minute = files_total / (session_length_hours * 60) if session_length_hours > 0 else 0
        files_per_hour = files_total / session_length_hours if session_length_hours > 0 else 0
        
        # Качество данных
        jpg_mov_ratio = jpg_count / mov_count if mov_count > 0 else (jpg_count if jpg_count > 0 else 0)
        error_rate = err_count / files_total if files_total > 0 else 0
        date_consistency = 1 if not err_count else 0  
        
        # Анализ интервалов
        gap_outliers = detect_gap_outliers(diffs) if diffs else 0
        activity_intensity = calculate_activity_intensity(files_per_hour)
        burst_activity = detect_burst_activity(diffs) if diffs else 0
        
        # AI метрики
        ai_consensus = 0.0
        ai_entropy = 0.0
        ai_confidence_gap = 0.0
        ai_second_place_gap = 0.0
        prediction_reliability = 0.0
        detection_quality = "unknown"
        ai_class_diversity = 0
        ai_uncertainty_level = "unknown"
        
        if ENABLE_AI and ai is not None and files_total > 0 and 'res' in locals():
            ai_consensus = calculate_ai_consensus(res.get("per_frame", []), ai_final_label)
            ai_entropy = calculate_ai_entropy(presence) if 'presence' in locals() else 0.0
            
            # Разрывы в уверенности
            if len(top_presence) >= 2:
                ai_confidence_gap = top_presence[0][1] - top_presence[1][1]
            if len(top_presence) >= 3:
                ai_second_place_gap = top_presence[1][1] - top_presence[2][1]
            
            prediction_reliability = calculate_prediction_reliability(
                ai_consensus, ai_confidence_gap, ai_entropy, ai_samples
            )
            detection_quality = calculate_detection_quality(
                ai_consensus, ai_entropy, ai_samples, float(ai_final_prob or 0)
            )
            ai_class_diversity = calculate_ai_class_diversity(presence) if 'presence' in locals() else 0
            ai_uncertainty_level = calculate_ai_uncertainty_level(ai_entropy)

        # Строка для общей базы
        if SERIES_DB_PATH:
            series_uid = hashlib.sha1(str(d.resolve()).encode("utf-8")).hexdigest()
            row = {
                "run_ts": run_ts,
                "root_path": str(root),
                "series_relpath": str(d.relative_to(root)),
                "series_dirname": d.name,
                "series_uid": series_uid,
                "date": day.strftime("%Y-%m-%d"),
                "weekday": weekday_name(day),
                "start_hhmm": start_hhmm,
                "end_hhmm": end_hhmm,
                "first_dt": mn.strftime("%Y-%m-%d %H:%M:%S") if times else "",
                "last_dt":  mx.strftime("%Y-%m-%d %H:%M:%S") if times else "",
                "duration_sec": int((mx - mn).total_seconds()) if times else "",
                "files_total": files_total,
                "jpg_count": jpg_count,
                "mov_count": mov_count,
                "errors_in_series": err_count,
                "min_gap_sec": int(gmin) if gmin is not None else "",
                "max_gap_sec": int(gmax) if gmax is not None else "",
                "mean_gap_sec": round(gmean, 3) if gmean is not None else "",
                "median_gap_sec": round(gmed, 3) if gmed is not None else "",
                "hour_of_day": hour_of_day,
                "is_weekend": is_weekend,
                "time_of_day_category": time_of_day_category,
                "session_length_hours": round(session_length_hours, 2) if session_length_hours else "",
                "files_per_minute": round(files_per_minute, 2) if files_per_minute else "",
                "files_per_hour": round(files_per_hour, 2) if files_per_hour else "",
                "jpg_mov_ratio": round(jpg_mov_ratio, 2) if jpg_mov_ratio else "",
                "error_rate": round(error_rate, 3) if error_rate else "",
                "date_consistency": date_consistency,
                "gap_outliers": gap_outliers,
                "activity_intensity": activity_intensity,
                "burst_activity": burst_activity,
            }
            if AI_ADD_COLUMNS:
                row.update({
                    "ai_model": ai_model_name,
                    "ai_samples": ai_samples if ai_samples else "",
                    "ai_top1_label": ai_top1_label,
                    "ai_top1_series_prob": ai_top1_prob,
                    "ai_top3": ai_topk_str,
                    "ai_counts_json": ai_counts_json,
                    "ai_final_label": ai_final_label,
                    "ai_final_prob": ai_final_prob,
                    "ai_consensus": round(ai_consensus, 3) if ai_consensus else "",
                    "ai_entropy": round(ai_entropy, 3) if ai_entropy else "",
                    "ai_confidence_gap": round(ai_confidence_gap, 3) if ai_confidence_gap else "",
                    "ai_second_place_gap": round(ai_second_place_gap, 3) if ai_second_place_gap else "",
                    "prediction_reliability": round(prediction_reliability, 3) if prediction_reliability else "",
                    "detection_quality": detection_quality,
                    "ai_class_diversity": ai_class_diversity,
                    "ai_uncertainty_level": ai_uncertainty_level
                })
            db_rows.append(row)

    # NoDate: не должно быть файлов с датой
    if CHECK_NODATE and nodate_files:
        for p in nodate_files:
            dt = meta_dt.get(p)
            if dt:
                details_rows.append([str(nodate_dir), p.name, dt.strftime("%Y-%m-%d %H:%M:%S"),
                                     "", "NODATE_HAS_DATE",
                                     "В NoDate оказался файл с датой — надо перенести"])
                total_errors += 1

    # Отчёты
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    sum_csv = (Path(REPORTS_DIR) if REPORTS_DIR else root) / f"series_check_summary_{ts}.csv"
    det_csv = (Path(REPORTS_DIR) if REPORTS_DIR else root) / f"series_check_details_{ts}.csv"

    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series_dir", "files", "errors", "min_time", "max_time"])
        for row in summary_rows:
            w.writerow(row)

    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series_dir", "file", "file_datetime", "series_interval", "error_code", "note"])
        for row in details_rows:
            w.writerow(row)

    # База серий (append)
    if SERIES_DB_PATH:
        write_series_db_rows(Path(SERIES_DB_PATH).resolve(), db_rows, with_bom=WRITE_DB_WITH_BOM)
        print(f"База серий пополнена: {SERIES_DB_PATH}")

    t1 = time.time()
    ok_dirs = sum(1 for row in summary_rows if row[2] == 0)
    print(f"Проверено папок: {len(per_dir_files)}. Без ошибок: {ok_dirs}. Ошибок всего: {total_errors}.")
    print(f"Сводка: {sum_csv}")
    print(f"Детали: {det_csv}")
    print(f"Готово за {t1 - t0:.1f} сек.")

if __name__ == "__main__":
    main()
