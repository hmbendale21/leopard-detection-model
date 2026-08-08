# Deploying LeopardEye

## What was broken

The original app tried to open a webcam window (`cv2.imshow`) in a
background process **on the server**, coordinated with the Flask app
through flag files (`.model_ready`, `.detect_trigger`, `.detect_stop`).
That only works on your own machine, where the server process and the
webcam/display are the same computer. On any hosted platform:

- There is no camera attached to the server.
- There is no display for `cv2.imshow` to draw into.
- Vercel specifically also can't run this: serverless functions are
  stateless/short-lived and can't host a persistent background
  subprocess, and the `torch` + `ultralytics` dependencies are far
  larger than Vercel's Python function size limit.

**Fix:** the browser now captures the visitor's own webcam
(`getUserMedia`) and posts individual frames to a new
`POST /api/detect/frame` endpoint. The server just scores each frame
and returns the bounding box/confidence — no camera or GUI window on
the server at all. This is the standard pattern for a web-hosted CV
demo and works identically on localhost and in production.

## Where to deploy

Because this still needs `torch` + `ultralytics` in memory, it needs a
host that runs a real, persistent Python process (not Vercel's
serverless functions). Good options, roughly easiest first:

- **Render** — "New Web Service" from your repo, it auto-detects the
  `Dockerfile`. Pick at least the 1 GB RAM tier.
- **Railway** — "New Project → Deploy from repo", it also picks up the
  `Dockerfile` automatically.
- **Fly.io** — `fly launch` in this folder, then `fly deploy`.
- **Hugging Face Spaces** (Docker SDK) — good free option for demos.
- Any VM / Cloud Run / ECS that can run a Docker container.

Locally, or on any of the above:

```bash
docker build -t leopardeye .
docker run -p 5000:5000 leopardeye
```

or without Docker:

```bash
pip install -r requirements.txt
python app.py
```

## One important browser requirement

`getUserMedia` (camera access) only works over **HTTPS**, or on
`http://localhost`. Every host listed above serves HTTPS by default,
so this isn't extra work — just don't expect the camera button to work
if you put the app behind plain HTTP on a custom domain.

## Notes

- `test_live_cam.py` is kept as-is for optional local/offline use
  (e.g. running detection over a saved video file from the command
  line). It's independent of the web app now and isn't part of the
  deploy path.
- `/healthz` returns `{"status": "ok"}` once the process is up, useful
  for platform health checks (it doesn't wait for models to finish
  loading — use `/api/status` for that).
