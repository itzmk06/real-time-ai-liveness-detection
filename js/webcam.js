// static/js/webcam.js
let socket = null;
const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const nonceEl = document.getElementById("nonce");
const challengeEl = document.getElementById("challenge");
const progressFill = document.getElementById("progressFill");
const avgEl = document.getElementById("avg");
const perframeEl = document.getElementById("perframe");
const framesEl = document.getElementById("frames");
const verdictEl = document.getElementById("verdict");
const embEl = document.getElementById("emb");

function fitCanvas() {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
  .then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      fitCanvas();
    };
    window.addEventListener("resize", fitCanvas);
  })
  .catch(err => {
    statusEl.innerText = "Camera error: " + err.message;
  });

function openWs() {
  // connect to same host/port serving page
  const base = window.location.host;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${base}/ws`);

  socket.onopen = () => {
    console.log("ws open");
    statusEl.innerText = "WebSocket connected";
  };

  socket.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      // handshake nonce
      if (d.nonce) {
        nonceEl.innerText = "nonce: " + d.nonce;
      }
      if (d.status) {
        statusEl.innerText = d.status;
      }
      if (d.challenge_message) {
        challengeEl.innerText = d.challenge_message;
      } else if (d.challenge_state) {
        challengeEl.innerText = d.challenge_state;
      }
      if (typeof d.progress !== "undefined") {
        progressFill.style.width = Math.max(0, Math.min(100, d.progress)) + "%";
      }
      if (typeof d.avg_live !== "undefined") {
        avgEl.innerText = "Avg Live: " + d.avg_live.toFixed(3);
      }
      if (typeof d.per_frame_live !== "undefined") {
        perframeEl.innerText = "Per-frame: " + d.per_frame_live.toFixed(3);
      }
      if (typeof d.frames_processed !== "undefined") {
        framesEl.innerText = "Frames: " + d.frames_processed;
      }
      if (d.final_verdict) {
        verdictEl.innerText = "Verdict: " + d.final_verdict;
      }
      if (typeof d.embedding_in_progress !== "undefined") {
        embEl.innerText = "Embedding: " + (d.embedding_in_progress ? "in progress" : "idle");
      }
    } catch (e) {
      console.warn("invalid ws msg", e);
    }
  };

  socket.onerror = (e) => {
    console.error("ws error", e);
    statusEl.innerText = "WebSocket error";
  };

  socket.onclose = () => {
    statusEl.innerText = "WebSocket closed — reconnecting in 2s...";
    setTimeout(openWs, 2000);
  };
}
openWs();

// send reduced-size frames periodically
const SEND_INTERVAL = 120; // ms (adjust per backend speed)
setInterval(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (video.readyState < 2) return;
  // draw a smaller canvas to limit bytes
  const c = document.createElement("canvas");
  const W = Math.max(160, Math.floor(video.videoWidth * 0.6));
  const H = Math.max(120, Math.floor(video.videoHeight * 0.6));
  c.width = W; c.height = H;
  const cctx = c.getContext("2d");
  cctx.drawImage(video, 0, 0, W, H);
  c.toBlob((blob) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result; // "data:image/jpeg;base64,..."
      socket.send(JSON.stringify({ frame: dataUrl }));
    };
    reader.readAsDataURL(blob);
  }, "image/jpeg", 0.7);
}, SEND_INTERVAL);
