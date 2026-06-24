from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app.py",
    "recommender.py",
    "requirements.txt",
    "render.yaml",
    "templates/index.html",
    "static/css/style.css",
    "static/js/app.js",
    "data/videos_cleaned.csv",
    "data/users_cleaned.csv",
    "data/interactions_cleaned.csv",
    "models/content_similarity.pkl",
    "models/video_indices.pkl",
    "models/item_similarity.pkl",
    "models/video_to_idx.pkl",
    "models/idx_to_video.pkl",
    "models/svd_model.pkl",
    "models/all_videos.pkl",
    "notebooks/08_production_reel_recommendation_engine.ipynb",
]

REQUIRED_HEADERS = {
    "data/videos_cleaned.csv": {
        "video_id",
        "author_id",
        "video_type",
        "upload_type",
        "video_duration",
        "music_id",
        "music_type",
        "tag",
    },
    "data/users_cleaned.csv": {
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
    },
    "data/interactions_cleaned.csv": {
        "user_id",
        "video_id",
        "play_time_ms",
        "duration_ms",
        "watch_ratio",
        "engagement_score",
        "interaction_score",
    },
}


def _read_header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return set(next(csv.reader(file)))


def validate_files() -> list[str]:
    errors = []
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing required file: {relative_path}")
    return errors


def validate_csv_headers() -> list[str]:
    errors = []
    for relative_path, required_columns in REQUIRED_HEADERS.items():
        path = ROOT / relative_path
        if not path.exists():
            continue

        header = _read_header(path)
        missing = sorted(required_columns - header)
        if missing:
            errors.append(f"{relative_path} missing columns: {missing}")

    return errors


def validate_notebook() -> list[str]:
    path = ROOT / "notebooks" / "08_production_reel_recommendation_engine.ipynb"
    if not path.exists():
        return []

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Production notebook is invalid JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        return ["Production notebook must use nbformat 4"]

    if not notebook.get("cells"):
        return ["Production notebook has no cells"]

    return []


def main() -> int:
    errors = []
    errors.extend(validate_files())
    errors.extend(validate_csv_headers())
    errors.extend(validate_notebook())

    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
