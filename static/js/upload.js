const form = document.getElementById("upload-form");
const input = document.getElementById("audio-files");
const status = document.getElementById("upload-status");

function errorMessage(response, fallback) {
  return response.json()
    .then((data) => data.detail || fallback)
    .catch(() => fallback);
}

function formatTrackDetails(track) {
  const details = [track.status];
  if (track.bpm) details.push(`${Math.round(track.bpm)} BPM`);
  if (track.key) details.push(`${track.key} ${track.key_scale || ""}`.trim());
  if (track.duration_seconds) details.push(`${track.duration_seconds.toFixed(1)} sec`);
  return details.join(" · ");
}

async function refreshPendingTracks() {
  const rows = Array.from(document.querySelectorAll("[data-track-row]"))
    .filter((row) => !["ready", "failed"].includes(row.dataset.trackStatus));
  if (!rows.length) return;

  await Promise.all(rows.map(async (row) => {
    try {
      const response = await fetch(`/api/tracks/${row.dataset.trackId}`);
      if (!response.ok) return;
      const track = await response.json();
      row.dataset.trackStatus = track.status;
      row.querySelector("[data-track-details]").textContent = formatTrackDetails(track);
      const checkbox = row.querySelector("input[name=track]");
      checkbox.disabled = track.status !== "ready";
      if (track.status === "ready" && selectAllTracks?.checked) checkbox.checked = true;
    } catch (_) {
      // A temporary polling failure should not interrupt analysis.
    }
  }));

  if (document.querySelector('[data-track-row][data-track-status="queued"], [data-track-row][data-track-status="processing"]')) {
    window.setTimeout(refreshPendingTracks, 2000);
  }
}

refreshPendingTracks();

async function deleteItem(url, message, row) {
  if (!window.confirm(message)) return;
  const button = row.querySelector("[data-delete-track], [data-delete-mix]");
  button.disabled = true;
  try {
    const response = await fetch(url, { method: "DELETE" });
    if (!response.ok) throw new Error(await errorMessage(response, "Could not delete item."));
    row.remove();
    syncSelectAll();
  } catch (error) {
    button.disabled = false;
    window.alert(error.message);
  }
}

document.addEventListener("click", (event) => {
  const trackButton = event.target.closest("[data-delete-track]");
  if (trackButton) {
    const row = trackButton.closest("li");
    deleteItem(
      `/api/tracks/${trackButton.dataset.deleteTrack}`,
      `Delete “${trackButton.dataset.name}”? This removes the uploaded audio and its analysis.`,
      row,
    );
    return;
  }

  const mixButton = event.target.closest("[data-delete-mix]");
  if (mixButton) {
    const row = mixButton.closest("li");
    deleteItem(
      `/api/mixes/${mixButton.dataset.deleteMix}`,
      `Delete Mix #${mixButton.dataset.deleteMix}? This removes the generated audio file.`,
      row,
    );
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = Array.from(input.files);
  if (!files.length) return;

  form.querySelector("button").disabled = true;
  status.textContent = `Uploading 0 of ${files.length}…`;
  try {
    for (const [index, file] of files.entries()) {
      const data = new FormData();
      data.append("file", file);
      const response = await fetch("/api/tracks", { method: "POST", body: data });
      if (!response.ok) {
        throw new Error(await errorMessage(response, `Could not upload ${file.name}`));
      }
      status.textContent = `Uploaded ${index + 1} of ${files.length}…`;
    }
    status.textContent = "Upload complete. Beat analysis is running in the background.";
    input.value = "";
    window.setTimeout(() => window.location.reload(), 800);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    form.querySelector("button").disabled = false;
  }
});

const mixForm = document.getElementById("mix-form");
const mixStatus = document.getElementById("mix-status");
const selectAllTracks = document.getElementById("select-all-tracks");
const generationProgress = document.getElementById("generation-progress");
let generationStartedAt = null;
let generationTimer = null;

function showGenerationProgress() {
  if (!generationProgress || !generationProgress.hidden) return;
  generationProgress.hidden = false;
  generationStartedAt = Date.now();
  const elapsed = generationProgress.querySelector("[data-generation-elapsed]");
  const stage = generationProgress.querySelector("[data-generation-stage]");
  const stages = [
    [0, "Analyzing the best transitions…"],
    [8, "Aligning beats and musical phrases…"],
    [18, "Blending levels and transitions…"],
    [35, "Rendering your final audio…"],
  ];
  generationTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - generationStartedAt) / 1000);
    elapsed.textContent = formatTime(seconds);
    stage.textContent = stages.filter(([at]) => seconds >= at).at(-1)[1];
  }, 1000);
}

function hideGenerationProgress() {
  if (!generationProgress) return;
  generationProgress.hidden = true;
  window.clearInterval(generationTimer);
}

function readyTrackCheckboxes() {
  return Array.from(document.querySelectorAll('input[name="track"]:not(:disabled)'));
}

function syncSelectAll() {
  if (!selectAllTracks) return;
  const checkboxes = readyTrackCheckboxes();
  const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;
  selectAllTracks.checked = checkboxes.length > 0 && selectedCount === checkboxes.length;
  selectAllTracks.indeterminate = selectedCount > 0 && selectedCount < checkboxes.length;
  selectAllTracks.disabled = checkboxes.length === 0;
  const selectionCount = document.querySelector("[data-selection-count]");
  if (selectionCount) selectionCount.textContent = `${selectedCount} selected`;
}

if (selectAllTracks) {
  selectAllTracks.addEventListener("change", () => {
    readyTrackCheckboxes().forEach((checkbox) => {
      checkbox.checked = selectAllTracks.checked;
    });
    syncSelectAll();
  });
  document.getElementById("track-list").addEventListener("change", (event) => {
    if (event.target.matches('input[name="track"]')) syncSelectAll();
  });
  syncSelectAll();
}

if (mixForm) {
  mixForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const trackIds = Array.from(mixForm.querySelectorAll("input[name=track]:checked"))
      .map((input) => Number(input.value));
    if (trackIds.length < 2) {
      mixStatus.textContent = "Select at least two ready tracks.";
      return;
    }

    const button = mixForm.querySelector('button[type="submit"]');
    button.disabled = true;
    mixStatus.textContent = "Queueing your mix…";
    showGenerationProgress();
    try {
      const response = await fetch("/api/mixes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_ids: trackIds,
          horn_rough_transitions: document.getElementById("horn-rough-transitions").checked,
        }),
      });
      if (!response.ok) {
        throw new Error(await errorMessage(response, "Could not generate the mix."));
      }
      const data = await response.json();
      mixStatus.textContent = "Mix is processing. This page will refresh when it is ready.";
      waitForMix(data.id);
    } catch (error) {
      mixStatus.textContent = error.message;
      button.disabled = false;
      hideGenerationProgress();
    }
  });
}

async function waitForMix(mixId) {
  try {
    const response = await fetch(`/api/mixes/${mixId}`);
    if (!response.ok) throw new Error("Could not check mix status.");
    const mix = await response.json();
    if (mix.status === "ready" || mix.status === "failed") {
      window.location.reload();
      return;
    }
    window.setTimeout(() => waitForMix(mixId), 2000);
  } catch (error) {
    mixStatus.textContent = error.message;
  }
}

async function refreshPendingMixes() {
  const rows = Array.from(document.querySelectorAll("[data-mix-row]"))
    .filter((row) => ["queued", "processing"].includes(row.dataset.mixStatus));
  if (!rows.length) return;
  try {
    const mixes = await Promise.all(rows.map(async (row) => {
      const response = await fetch(`/api/mixes/${row.dataset.mixId}`);
      if (!response.ok) throw new Error("Could not check mix status.");
      return response.json();
    }));
    if (mixes.some((mix) => ["ready", "failed"].includes(mix.status))) {
      window.location.reload();
      return;
    }
  } catch (_) {
    // Keep the visible progress state and retry after a temporary failure.
  }
  window.setTimeout(refreshPendingMixes, 2000);
}

refreshPendingMixes();

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

document.querySelectorAll("[data-player]").forEach((player) => {
  const audio = player.querySelector("[data-audio]");
  const toggle = player.querySelector("[data-play]");
  const progress = player.querySelector("[data-progress]");
  const current = player.querySelector("[data-current]");
  const waveform = player.querySelector("[data-waveform]");
  const waveformBars = player.querySelector("[data-waveform-bars]");
  const playerName = player.dataset.playerName || "mix";

  fetch(waveform.dataset.waveformUrl)
    .then((response) => {
      if (!response.ok) throw new Error("Waveform unavailable");
      return response.json();
    })
    .then(({ peaks, bpm }) => {
      waveformBars.replaceChildren(...peaks.map((peak) => {
        const bar = document.createElement("span");
        bar.style.height = `${Math.max(5, peak * 100)}%`;
        return bar;
      }));
      if (bpm) waveform.dataset.bpm = bpm;
      const duration = Number(waveform.dataset.duration);
      if (bpm && duration) {
        const barWidth = ((240 / bpm) / duration) * 100;
        waveformBars.style.setProperty("--bar-width", `${barWidth}%`);
        waveformBars.classList.add("has-tempo-grid");
      }
    })
    .catch(() => waveform.classList.add("waveform-fallback"));

  toggle.addEventListener("click", async () => {
    if (audio.paused) {
      document.querySelectorAll("[data-audio]").forEach((otherAudio) => {
        if (otherAudio !== audio) otherAudio.pause();
      });
      await audio.play();
    } else {
      audio.pause();
    }
  });

  audio.addEventListener("play", () => {
    toggle.textContent = "❚❚";
    toggle.setAttribute("aria-label", `Pause ${playerName}`);
  });
  audio.addEventListener("pause", () => {
    toggle.textContent = "▶";
    toggle.setAttribute("aria-label", `Play ${playerName}`);
  });
  audio.addEventListener("timeupdate", () => {
    const percent = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
    progress.value = percent;
    waveform.style.setProperty("--played", `${percent}%`);
    current.textContent = formatTime(audio.currentTime);
  });
  audio.addEventListener("ended", () => {
    progress.value = 0;
    waveform.style.setProperty("--played", "0%");
  });
  progress.addEventListener("input", () => {
    if (audio.duration) audio.currentTime = (Number(progress.value) / 100) * audio.duration;
  });
});
