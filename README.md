# Reel Recommendation Engine

Production-oriented hybrid recommendation API for an e-commerce reels feed.

The system recommends the next reels for a user based on the reel they are currently watching. It combines:

- Content-based similarity from reel metadata
- Item-item collaborative filtering from user-reel interactions
- SVD matrix factorization for personalization
- Popularity fallback for cold-start and sparse cases

## API

### Health

```http
GET /health
```

Returns model/data status, matrix shapes, cache size, and hybrid weights.

### Recommendations

```http
GET /recommend?user_id=0&video_id=1527&top_n=10
```

Returns hybrid recommendations with model scores and reel metadata.

### Trending

```http
GET /trending?top_n=20
```

Returns globally popular reels. Use this for logged-out users, onboarding, and fallback feeds.

### User Status

```http
GET /user/<user_id>
```

Returns whether the user exists and whether personalized or cold-start mode should be used.

### Video Status

```http
GET /video/<video_id>
```

Returns whether the reel exists and whether it is indexed by content/collaborative models.

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## Validation

Run lightweight project checks without loading the heavy model artifacts:

```bash
python scripts/validate_project.py
python -m py_compile app.py recommender.py scripts/validate_project.py
```

## Production Notes

The API loads large pickle artifacts at startup. Use Gunicorn with a longer timeout:

```bash
gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
```

The SVD predictions are cached per user in memory, so the first request for a user can be slower and repeated requests are faster.

## E-Commerce Integration

Call `/recommend` from the reels section whenever a user is watching a reel. The response includes `video_id`, model scores, recommendation type, and video metadata. The e-commerce app can join `video_id` to its own product-tag table.

For a stronger product-aware model, add these events to the dataset:

- product click from reel
- add to cart
- purchase
- wishlist
- product category
- price bucket
- stock availability
- margin or revenue

Once those fields exist, use this hybrid engine as candidate generation and train a learning-to-rank model to rerank candidates for conversion and revenue.

## Important

No recommendation model can guarantee 100% accuracy. Production readiness comes from reliable fallbacks, fast response time, offline metrics, online A/B tests, monitoring, and continuous retraining.
