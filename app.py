from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from recommender import (
    MAX_TOP_N,
    get_hybrid_recommendations,
    get_popular_recommendations,
    get_system_status,
    get_user_status,
    get_video_status,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "success",
        "message": "Reel recommendation API is running",
        "system": get_system_status(),
    }), 200


@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    video_id = request.args.get("video_id", type=int)
    top_n = request.args.get("top_n", default=10, type=int)

    if user_id is None:
        return _error(
            "user_id is required. Example: /recommend?user_id=0&video_id=1527",
            400,
        )

    if video_id is None:
        return _error(
            "video_id is required. Example: /recommend?user_id=0&video_id=1527",
            400,
        )

    try:
        recommendations = get_hybrid_recommendations(
            user_id=user_id,
            video_id=video_id,
            top_n=top_n,
        )
    except Exception:
        logger.exception("Recommendation request failed")
        return _error("Internal server error while generating recommendations", 500)

    return jsonify({
        "status": "success",
        "user_id": user_id,
        "video_id": video_id,
        "requested_top_n": top_n,
        "max_top_n": MAX_TOP_N,
        "count": len(recommendations),
        "recommendations": recommendations,
    }), 200


@app.route("/trending", methods=["GET"])
def trending():
    top_n = request.args.get("top_n", default=20, type=int)

    try:
        recommendations = get_popular_recommendations(top_n=top_n)
    except Exception:
        logger.exception("Trending request failed")
        return _error("Internal server error while loading trending reels", 500)

    return jsonify({
        "status": "success",
        "requested_top_n": top_n,
        "max_top_n": MAX_TOP_N,
        "count": len(recommendations),
        "recommendations": recommendations,
    }), 200


@app.route("/top", methods=["GET"])
def top_recommendations():
    return trending()


@app.route("/user/<int:user_id>", methods=["GET"])
def user_info(user_id: int):
    return jsonify({
        "status": "success",
        "user": get_user_status(user_id),
    }), 200


@app.route("/video/<int:video_id>", methods=["GET"])
def video_info(video_id: int):
    return jsonify({
        "status": "success",
        "video": get_video_status(video_id),
    }), 200


@app.errorhandler(404)
def not_found(_error_obj):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "GET /",
            "GET /health",
            "GET /recommend?user_id=<int>&video_id=<int>&top_n=<int>",
            "GET /trending?top_n=<int>",
            "GET /top?top_n=<int>",
            "GET /user/<user_id>",
            "GET /video/<video_id>",
        ],
    }), 404


@app.errorhandler(405)
def method_not_allowed(_error_obj):
    return _error("Method not allowed. This API currently accepts GET requests.", 405)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
