(function () {
  const root = document.querySelector("[data-action-token]");
  if (!root) {
    return;
  }

  const actionToken = root.dataset.actionToken || "";
  const statusUrl = root.dataset.statusUrl || "";
  let pollingTimer = 0;
  let currentDelay = 0;

  function setFormError(form, message) {
    const target = form.querySelector("[data-form-error]");
    if (!target) {
      return;
    }
    target.textContent = message;
    target.hidden = false;
  }

  function clearFormError(form) {
    const target = form.querySelector("[data-form-error]");
    if (target) {
      target.hidden = true;
    }
  }

  async function submitEnhancedForm(event) {
    const form = event.target.closest("[data-enhance-submit]");
    if (!form) {
      return;
    }
    event.preventDefault();
    clearFormError(form);
    const body = new URLSearchParams(new FormData(form));
    body.delete("response_format");
    try {
      const response = await fetch(form.action, {
        method: form.method || "POST",
        headers: {
          "accept": "application/json",
          "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
          "x-control-plane-token": actionToken,
        },
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Run could not start.");
      }
      if (payload.status_url) {
        window.location.assign(payload.status_url);
      }
    } catch (error) {
      setFormError(form, error.message || "Run could not start.");
    }
  }

  function stopPolling() {
    if (pollingTimer) {
      window.clearInterval(pollingTimer);
      pollingTimer = 0;
      currentDelay = 0;
    }
  }

  function startPolling(active) {
    if (!statusUrl || document.hidden) {
      return;
    }
    const delay = active ? 2000 : 10000;
    if (pollingTimer && currentDelay === delay) {
      return;
    }
    stopPolling();
    currentDelay = delay;
    pollingTimer = window.setInterval(pollStatus, delay);
  }

  function statusLabel(payload) {
    if (payload.active) {
      return "Running";
    }
    if (payload.next_resume_stage) {
      return "Resumable";
    }
    const stages = Array.isArray(payload.stages) ? payload.stages : [];
    const failed = stages.some((stage) => stage.status === "failed");
    if (failed) {
      return "Failed";
    }
    const succeeded = stages.length > 0 && stages.every((stage) => stage.status === "succeeded" || stage.status === "skipped");
    if (succeeded) {
      return "Succeeded";
    }
    const planned = stages.length > 0 && stages.every((stage) => (
      stage.status === "planned" || stage.status === "succeeded" || stage.status === "skipped"
    ));
    if (planned) {
      return "Planned";
    }
    return "Not started";
  }

  function setStatusBadge(badge, label) {
    badge.textContent = label;
    badge.classList.remove(
      "status-not-started",
      "status-running",
      "status-succeeded",
      "status-skipped",
      "status-planned",
      "status-failed",
      "status-resumable",
    );
    const key = label.toLowerCase().replace(" ", "-");
    badge.classList.add(`status-${key}`);
  }

  async function pollStatus() {
    if (!statusUrl || document.hidden) {
      stopPolling();
      return;
    }
    try {
      const response = await fetch(statusUrl, { headers: { "accept": "application/json" } });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      const label = statusLabel(payload);
      document.querySelectorAll("[data-run-summary-status]").forEach((badge) => setStatusBadge(badge, label));
      const nextResume = document.querySelector("[data-next-resume-stage]");
      if (nextResume) {
        nextResume.textContent = payload.next_resume_stage || "None";
      }
      root.dataset.runActive = payload.active ? "true" : "false";
      startPolling(Boolean(payload.active));
    } catch (_error) {
      return;
    }
  }

  function escapeText(value) {
    const span = document.createElement("span");
    span.textContent = value;
    return span.innerHTML;
  }

  async function renderTextPreview(preview_url, label, type) {
    const response = await fetch(preview_url, { headers: { "accept": type === "json" ? "application/json" : "text/plain" } });
    const text = await response.text();
    return `
      <div class="preview-heading">
        <h2>${escapeText(label)}</h2>
        <a class="icon-button" href="${preview_url}" target="_blank" rel="noreferrer" aria-label="Open ${escapeText(label)} in a new tab" data-tooltip="Open">Open</a>
      </div>
      <pre>${escapeText(text)}</pre>
    `;
  }

  async function renderPreview(button) {
    const pane = document.querySelector("[data-preview-pane]");
    if (!pane) {
      return;
    }
    const preview_url = button.dataset.previewUrl;
    const previewType = button.dataset.previewType;
    const label = button.dataset.previewLabel || "Artifact preview";
    if (!preview_url) {
      return;
    }
    if (previewType === "markdown" || previewType === "json") {
      pane.innerHTML = await renderTextPreview(preview_url, label, previewType);
      return;
    }
    const heading = `
      <div class="preview-heading">
        <h2>${escapeText(label)}</h2>
        <a class="icon-button" href="${preview_url}" target="_blank" rel="noreferrer" aria-label="Open ${escapeText(label)} in a new tab" data-tooltip="Open">Open</a>
      </div>
    `;
    if (previewType === "audio") {
      pane.innerHTML = `${heading}<audio controls src="${preview_url}"></audio>`;
    } else if (previewType === "video") {
      pane.innerHTML = `${heading}<video controls src="${preview_url}"></video>`;
    } else if (previewType === "deck-html") {
      pane.innerHTML = `${heading}<iframe sandbox="allow-scripts" class="preview-frame" title="${escapeText(label)}" src="${preview_url}"></iframe>`;
    } else if (previewType === "document") {
      pane.innerHTML = `${heading}<object class="preview-object" data="${preview_url}"></object>`;
    }
  }

  async function copyValue(button) {
    const value = button.dataset.copyValue || "";
    if (!value) {
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      button.dataset.tooltip = "Copied";
    } catch (_error) {
      const field = document.createElement("textarea");
      field.value = value;
      document.body.appendChild(field);
      field.select();
      document.execCommand("copy");
      field.remove();
      button.dataset.tooltip = "Copied";
    }
  }

  document.addEventListener("submit", submitEnhancedForm);
  document.addEventListener("click", (event) => {
    const previewButton = event.target.closest("[data-preview-button]");
    if (previewButton) {
      event.preventDefault();
      renderPreview(previewButton);
      return;
    }
    const copyButton = event.target.closest("[data-copy-value]");
    if (copyButton) {
      event.preventDefault();
      copyValue(copyButton);
      return;
    }
    const refreshButton = event.target.closest("[data-refresh-button]");
    if (refreshButton) {
      event.preventDefault();
      pollStatus();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopPolling();
    } else {
      startPolling(root.dataset.runActive === "true");
    }
  });
  window.addEventListener("pagehide", stopPolling);
  window.addEventListener("beforeunload", stopPolling);

  startPolling(root.dataset.runActive === "true");
})();
