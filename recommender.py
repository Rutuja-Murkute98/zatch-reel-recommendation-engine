from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

CONTENT_WEIGHT = 0.35
COLLABORATIVE_WEIGHT = 0.35
SVD_WEIGHT = 0.30
DEFAULT_CANDIDATE_POOL = 300
MAX_TOP_N = 50


def _load_pickle(name: str) -> Any:
    with (MODEL_DIR / name).open("rb") as file:
        return pickle.load(file)


def _load_models() -> dict[str, Any]:
    logger.info("Loading recommendation model artifacts")
    return {
        "content_similarity": _load_pickle("content_similarity.pkl"),
        "video_indices": _load_pickle("video_indices.pkl"),
        "item_similarity": _load_pickle("item_similarity.pkl"),
        "video_to_idx": _load_pickle("video_to_idx.pkl"),
        "idx_to_video": _load_pickle("idx_to_video.pkl"),
        "svd_model": _load_pickle("svd_model.pkl"),
        "all_videos": _load_pickle("all_videos.pkl"),
    }


def _load_datasets() -> dict[str, pd.DataFrame]:
    logger.info("Loading recommendation datasets")
    return {
        "videos": pd.read_csv(DATA_DIR / "videos_cleaned.csv"),
        "users": pd.read_csv(DATA_DIR / "users_cleaned.csv"),
        "interactions": pd.read_csv(DATA_DIR / "interactions_cleaned.csv"),
    }


try:
    _MODELS = _load_models()
    _DATA = _load_datasets()
    MODELS_LOADED = True
except Exception:
    MODELS_LOADED = False
    logger.exception("Failed to load recommendation system")
    raise


content_similarity: np.ndarray = _MODELS["content_similarity"]
video_indices = _MODELS["video_indices"]
item_similarity: np.ndarray = _MODELS["item_similarity"]
video_to_idx: dict[int, int] = _MODELS["video_to_idx"]
idx_to_video: dict[int, int] = _MODELS["idx_to_video"]
svd_model = _MODELS["svd_model"]
all_videos = _MODELS["all_videos"]

videos_df = _DATA["videos"]
users_df = _DATA["users"]
interactions_df = _DATA["interactions"]

ALL_USER_IDS = set(users_df["user_id"].astype(int).tolist())
ALL_VIDEO_IDS = set(videos_df["video_id"].astype(int).tolist())

VIDEO_METADATA = (
    videos_df.set_index("video_id")
    .replace({np.nan: None})
    .to_dict(orient="index")
)

USER_SEEN_VIDEOS = {
    int(user_id): set(group["video_id"].astype(int).tolist())
    for user_id, group in interactions_df.groupby("user_id", sort=False)
}

POPULAR_VIDEOS = (
    interactions_df.groupby("video_id")["interaction_score"]
    .mean()
    .sort_values(ascending=False)
    .index.astype(int)
    .tolist()
)

logger.info(
    "Recommendation system ready: users=%s videos=%s interactions=%s",
    len(ALL_USER_IDS),
    len(ALL_VIDEO_IDS),
    len(interactions_df),
)


def _clamp_top_n(top_n: int | None) -> int:
    if top_n is None:
        return 10
    return max(1, min(int(top_n), MAX_TOP_N))


def _normalize_scores(recommendations: list[tuple[int, float]]) -> dict[int, float]:
    if not recommendations:
        return {}

    values = [score for _, score in recommendations]
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {int(video_id): 1.0 for video_id, _ in recommendations}

    return {
        int(video_id): float((score - min_score) / (max_score - min_score))
        for video_id, score in recommendations
    }


def _seen_videos_for_user(user_id: int) -> set[int]:
    return USER_SEEN_VIDEOS.get(int(user_id), set())


def _video_metadata(video_id: int) -> dict[str, Any]:
    metadata = VIDEO_METADATA.get(int(video_id), {})
    return {
        "author_id": metadata.get("author_id"),
        "video_type": metadata.get("video_type"),
        "upload_type": metadata.get("upload_type"),
        "video_duration": metadata.get("video_duration"),
        "music_id": metadata.get("music_id"),
        "music_type": metadata.get("music_type"),
        "tag": metadata.get("tag"),
    }


def _format_recommendation(
    video_id: int,
    content_score: float = 0.0,
    collaborative_score: float = 0.0,
    svd_score: float = 0.0,
    hybrid_score: float = 0.0,
    recommendation_type: str = "hybrid",
) -> dict[str, Any]:
    return {
        "video_id": int(video_id),
        "content_score": round(float(content_score), 4),
        "collaborative_score": round(float(collaborative_score), 4),
        "svd_score": round(float(svd_score), 4),
        "hybrid_score": round(float(hybrid_score), 4),
        "recommendation_type": recommendation_type,
        "video_metadata": _video_metadata(video_id),
    }


def get_content_recommendations(
    video_id: int,
    top_n: int = DEFAULT_CANDIDATE_POOL,
) -> list[tuple[int, float]]:
    if video_id not in video_indices:
        logger.info("Video %s not found in content model", video_id)
        return []

    idx = int(video_indices[video_id])
    scores = sorted(
        enumerate(content_similarity[idx]),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (int(videos_df.iloc[video_idx]["video_id"]), float(score))
        for video_idx, score in scores[1 : top_n + 1]
    ]


def get_collaborative_recommendations(
    video_id: int,
    top_n: int = DEFAULT_CANDIDATE_POOL,
) -> list[tuple[int, float]]:
    if video_id not in video_to_idx:
        logger.info("Video %s not found in collaborative model", video_id)
        return []

    idx = int(video_to_idx[video_id])
    scores = sorted(
        enumerate(item_similarity[idx]),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (int(idx_to_video[item_idx]), float(score))
        for item_idx, score in scores[1 : top_n + 1]
    ]


@lru_cache(maxsize=10_000)
def _get_svd_scores_for_user(user_id: int) -> tuple[tuple[int, float], ...]:
    scores: list[tuple[int, float]] = []

    for video_id in all_videos:
        try:
            prediction = svd_model.predict(int(user_id), int(video_id))
        except Exception:
            continue
        scores.append((int(video_id), float(prediction.est)))

    scores.sort(key=lambda item: item[1], reverse=True)
    logger.info("Built SVD cache for user=%s videos=%s", user_id, len(scores))
    return tuple(scores)


def get_svd_recommendations(
    user_id: int,
    top_n: int = DEFAULT_CANDIDATE_POOL,
) -> list[tuple[int, float]]:
    return list(_get_svd_scores_for_user(int(user_id))[:top_n])


def get_popular_recommendations(
    top_n: int = 20,
    seen_videos: set[int] | None = None,
    exclude_video: int | None = None,
    recommendation_type: str = "trending",
) -> list[dict[str, Any]]:
    top_n = _clamp_top_n(top_n)
    seen_videos = seen_videos or set()
    results: list[dict[str, Any]] = []

    for video_id in POPULAR_VIDEOS:
        if video_id in seen_videos:
            continue
        if exclude_video is not None and video_id == exclude_video:
            continue

        results.append(
            _format_recommendation(
                video_id,
                recommendation_type=recommendation_type,
            )
        )

        if len(results) == top_n:
            break

    return results


def get_hybrid_recommendations(
    user_id: int,
    video_id: int,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    top_n = _clamp_top_n(top_n)
    user_id = int(user_id)
    video_id = int(video_id)

    seen_videos = _seen_videos_for_user(user_id)
    is_new_user = user_id not in ALL_USER_IDS
    is_unknown_video = video_id not in ALL_VIDEO_IDS

    logger.info(
        "Hybrid request user=%s video=%s top_n=%s new_user=%s unknown_video=%s",
        user_id,
        video_id,
        top_n,
        is_new_user,
        is_unknown_video,
    )

    if is_new_user and is_unknown_video:
        return get_popular_recommendations(
            top_n=top_n,
            recommendation_type="popularity_cold_start",
        )

    content_recs = get_content_recommendations(video_id, top_n=DEFAULT_CANDIDATE_POOL)
    collaborative_recs = get_collaborative_recommendations(
        video_id,
        top_n=DEFAULT_CANDIDATE_POOL,
    )
    svd_recs = get_svd_recommendations(user_id, top_n=DEFAULT_CANDIDATE_POOL)

    content_scores = _normalize_scores(content_recs)
    collaborative_scores = _normalize_scores(collaborative_recs)
    svd_scores = _normalize_scores(svd_recs)

    candidate_video_ids = (
        set(content_scores)
        | set(collaborative_scores)
        | set(svd_scores)
    )
    candidate_video_ids.discard(video_id)
    candidate_video_ids -= seen_videos

    recommendations: list[dict[str, Any]] = []

    for candidate_video_id in candidate_video_ids:
        content_score = content_scores.get(candidate_video_id, 0.0)
        collaborative_score = collaborative_scores.get(candidate_video_id, 0.0)
        svd_score = svd_scores.get(candidate_video_id, 0.0)
        hybrid_score = (
            CONTENT_WEIGHT * content_score
            + COLLABORATIVE_WEIGHT * collaborative_score
            + SVD_WEIGHT * svd_score
        )

        recommendations.append(
            _format_recommendation(
                candidate_video_id,
                content_score=content_score,
                collaborative_score=collaborative_score,
                svd_score=svd_score,
                hybrid_score=hybrid_score,
                recommendation_type="hybrid",
            )
        )

    recommendations.sort(key=lambda item: item["hybrid_score"], reverse=True)

    if len(recommendations) < top_n:
        already_selected = {
            item["video_id"] for item in recommendations
        } | seen_videos | {video_id}

        padding = get_popular_recommendations(
            top_n=top_n - len(recommendations),
            seen_videos=already_selected,
            exclude_video=video_id,
            recommendation_type="popularity_fallback",
        )
        recommendations.extend(padding)

    return recommendations[:top_n]


def get_user_status(user_id: int) -> dict[str, Any]:
    user_id = int(user_id)
    exists = user_id in ALL_USER_IDS
    return {
        "user_id": user_id,
        "exists": exists,
        "seen_video_count": len(_seen_videos_for_user(user_id)),
        "recommendation_mode": "personalized" if exists else "cold_start",
    }


def get_video_status(video_id: int) -> dict[str, Any]:
    video_id = int(video_id)
    return {
        "video_id": video_id,
        "exists": video_id in ALL_VIDEO_IDS,
        "content_indexed": video_id in video_indices,
        "collaborative_indexed": video_id in video_to_idx,
        "video_metadata": _video_metadata(video_id),
    }


def get_system_status() -> dict[str, Any]:
    cache_info = _get_svd_scores_for_user.cache_info()
    return {
        "models_loaded": MODELS_LOADED,
        "total_users": len(ALL_USER_IDS),
        "total_videos": len(ALL_VIDEO_IDS),
        "total_interactions": len(interactions_df),
        "content_similarity_shape": list(content_similarity.shape),
        "item_similarity_shape": list(item_similarity.shape),
        "svd_cache_size": cache_info.currsize,
        "max_top_n": MAX_TOP_N,
        "weights": {
            "content": CONTENT_WEIGHT,
            "collaborative": COLLABORATIVE_WEIGHT,
            "svd": SVD_WEIGHT,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = get_hybrid_recommendations(user_id=0, video_id=1527, top_n=10)
    for recommendation in sample:
        print(recommendation)
