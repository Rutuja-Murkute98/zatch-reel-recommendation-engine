const resultsDiv = document.getElementById("results");
const recommendBtn = document.getElementById("recommendBtn");
const trendingBtn = document.getElementById("trendingBtn");
const healthBtn = document.getElementById("healthBtn");

function setLoading(message) {
  resultsDiv.innerHTML = `<div class="panel"><h2>${message}</h2></div>`;
}

function setError(message) {
  resultsDiv.innerHTML = `<div class="panel error"><h2>${message}</h2></div>`;
}

function metadataText(metadata) {
  if (!metadata) {
    return "";
  }

  const parts = [
    metadata.upload_type ? `upload: ${metadata.upload_type}` : "",
    metadata.tag !== null && metadata.tag !== undefined ? `tag: ${metadata.tag}` : "",
    metadata.video_duration ? `duration: ${Math.round(metadata.video_duration)}ms` : "",
  ].filter(Boolean);

  return parts.join(" - ");
}

function formatShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return "Not available";
  }
  return shape.join(" x ");
}

function renderRecommendationTable(title, data) {
  if (!data.recommendations || data.recommendations.length === 0) {
    setError("No recommendations found.");
    return;
  }

  const rows = data.recommendations
    .map((item, index) => {
      const meta = metadataText(item.video_metadata);
      return `
        <tr>
          <td>${index + 1}</td>
          <td><strong>${item.video_id}</strong><span>${meta}</span></td>
          <td>${item.hybrid_score}</td>
          <td>${item.content_score}</td>
          <td>${item.collaborative_score}</td>
          <td>${item.svd_score}</td>
          <td>${item.recommendation_type}</td>
        </tr>
      `;
    })
    .join("");

  resultsDiv.innerHTML = `
    <div class="panel">
      <h2>${title}</h2>
      <p>Returned ${data.count} results. API caps responses at ${data.max_top_n || 50} items.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Reel</th>
              <th>Hybrid</th>
              <th>Content</th>
              <th>CF</th>
              <th>SVD</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  `;
}

async function requestJson(url) {
  const response = await fetch(url);
  const data = await response.json();

  if (!response.ok || data.status !== "success") {
    throw new Error(data.message || "Request failed");
  }

  return data;
}

async function getRecommendations() {
  const userId = document.getElementById("user_id").value.trim();
  const videoId = document.getElementById("video_id").value.trim();
  const topN = document.getElementById("top_n").value.trim() || "10";

  if (!userId || !videoId) {
    setError("Enter both User ID and Current Reel / Video ID.");
    return;
  }

  recommendBtn.disabled = true;
  setLoading("Generating recommendations...");

  try {
    const params = new URLSearchParams({
      user_id: userId,
      video_id: videoId,
      top_n: topN,
    });
    const data = await requestJson(`/recommend?${params.toString()}`);
    renderRecommendationTable(
      `Recommendations for user ${data.user_id} watching reel ${data.video_id}`,
      data
    );
  } catch (error) {
    setError(error.message);
  } finally {
    recommendBtn.disabled = false;
  }
}

async function getTrending() {
  trendingBtn.disabled = true;
  setLoading("Loading trending reels...");

  try {
    const topN = document.getElementById("top_n").value.trim() || "20";
    const data = await requestJson(`/trending?top_n=${encodeURIComponent(topN)}`);
    renderRecommendationTable("Trending reels", data);
  } catch (error) {
    setError(error.message);
  } finally {
    trendingBtn.disabled = false;
  }
}

async function getHealth() {
  healthBtn.disabled = true;
  setLoading("Checking model health...");

  try {
    const data = await requestJson("/health");
    const system = data.system;
    resultsDiv.innerHTML = `
      <div class="panel">
        <h2>${data.message}</h2>
        <p>All model files and datasets are loaded. The recommendation API is ready to use.</p>
        <div class="metrics">
          <div><strong>${system.total_users.toLocaleString()}</strong><span>Users</span></div>
          <div><strong>${system.total_videos.toLocaleString()}</strong><span>Videos</span></div>
          <div><strong>${system.total_interactions.toLocaleString()}</strong><span>Interactions</span></div>
          <div><strong>${system.svd_cache_size.toLocaleString()}</strong><span>SVD cached users</span></div>
        </div>
        <div class="status-grid">
          <div>
            <span>Models loaded</span>
            <strong>${system.models_loaded ? "Yes" : "No"}</strong>
          </div>
          <div>
            <span>Content similarity matrix</span>
            <strong>${formatShape(system.content_similarity_shape)}</strong>
          </div>
          <div>
            <span>Collaborative matrix</span>
            <strong>${formatShape(system.item_similarity_shape)}</strong>
          </div>
          <div>
            <span>Hybrid weights</span>
            <strong>Content ${system.weights.content}, CF ${system.weights.collaborative}, SVD ${system.weights.svd}</strong>
          </div>
        </div>
      </div>
    `;
  } catch (error) {
    setError(error.message);
  } finally {
    healthBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  recommendBtn.addEventListener("click", getRecommendations);
  trendingBtn.addEventListener("click", getTrending);
  healthBtn.addEventListener("click", getHealth);
  getHealth();
});
