# AdLens

**Multimodal Video Advertisement Segment Detection System**

AdLens detects advertising segments in YouTube videos and local video files using a pipeline of ASR, OCR, visual classification, scene detection, and signal fusion. It returns a machine-readable timeline of every ad segment with timestamps, ad type, confidence, brand, description, and supporting evidence.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Environment Variables](#environment-variables)
5. [Running the API](#running-the-api)
6. [API Usage](#api-usage)
7. [Web Viewer](#web-viewer)
8. [Running the Test Set](#running-the-test-set)
9. [Ground Truth Annotation](#ground-truth-annotation)
10. [Running Evaluation](#running-evaluation)
11. [Running Tests](#running-tests)
12. [Instagram Reels](#instagram-reels)
13. [Architecture](#architecture)
14. [Known Limitations](#known-limitations)

---

## Project Overview

AdLens processes videos through 8 sequential stages:

```
Video URL / File → Ingestion → ASR → Scene Detection → Frame Sampling
                → OCR → Visual Classification → Signal Fusion → Segmentation
                → JSON Output
```

Each stage contributes evidence (audio signals, OCR text, visual classifications, scene cuts) that is fused into a per-window advertisement score, then segmented into discrete ad intervals.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.14 supported |
| FFmpeg | Any recent | Must be on system PATH |
| OpenAI API key | — | Optional — mock mode works without it |

### Install FFmpeg (Windows)

```powershell
winget install Gyan.FFmpeg
# Then restart your terminal
```

Verify: `ffmpeg -version`

---

## Installation

```powershell
# 1. Clone / enter the project
cd d:\AdLens

# 2. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
Copy-Item .env.example .env
# Edit .env — see next section
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key. Required for production providers. |
| `ADLENS_ASR_PROVIDER` | `mock` | `whisper_api` / `faster_whisper` / `mock` |
| `ADLENS_VISION_PROVIDER` | `mock` | `gpt4o_mini` / `mock` |
| `ADLENS_SAMPLING_STRATEGY` | `adaptive` | `adaptive` / `uniform` / `keyframe` |
| `ADLENS_SAMPLE_FPS` | `1.0` | FPS for uniform strategy |
| `ADLENS_MAX_FRAMES_PER_VIDEO` | `300` | Hard cap (cost guard) |
| `ADLENS_SCENE_THRESHOLD` | `27.0` | PySceneDetect sensitivity |
| `ADLENS_AD_SCORE_THRESHOLD` | `0.40` | Minimum fused score to flag ad |
| `ADLENS_MAX_GAP_S` | `3.0` | Gap tolerance for segment merging |
| `ADLENS_FUSION_AUDIO_WEIGHT` | `0.35` | Audio signal weight |
| `ADLENS_FUSION_VISUAL_WEIGHT` | `0.30` | Visual signal weight |
| `ADLENS_FUSION_OCR_WEIGHT` | `0.20` | OCR signal weight |
| `ADLENS_FUSION_SCENE_WEIGHT` | `0.15` | Scene signal weight |
| `ADLENS_WORK_DIR` | `./workdir` | Temp files directory |

### Development mode (no API key)

```ini
ADLENS_ASR_PROVIDER=mock
ADLENS_VISION_PROVIDER=mock
```

The system runs end-to-end with deterministic stubs. Results will NOT be accurate — this is for pipeline testing only.

### Production mode

```ini
OPENAI_API_KEY=sk-...
ADLENS_ASR_PROVIDER=whisper_api
ADLENS_VISION_PROVIDER=gpt4o_mini
ADLENS_SAMPLING_STRATEGY=adaptive
```

### Local AI mode (no API cost)

```ini
ADLENS_ASR_PROVIDER=faster_whisper   # downloads ~150 MB model on first run
ADLENS_VISION_PROVIDER=mock          # vision requires OpenAI key
```

---

## Running the API

```powershell
# Option 1: via main.py
python main.py

# Option 2: via uvicorn directly
uvicorn adlens.api.main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
#   http://localhost:8000/docs    (Swagger UI)
#   http://localhost:8000/redoc   (ReDoc)
#   http://localhost:8000/health  (health check)
#   http://localhost:8000/viewer  (web UI)
```

---

## API Usage

### Analyze a YouTube video

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=ujFWRFYLGjY"}'
```

### Analyze a local file (e.g. Instagram Reel)

```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "video=@/path/to/reel.mp4"
```

### Response format

```json
{
  "source": {
    "url": "https://...",
    "platform": "youtube",
    "kind": "vod",
    "duration_s": 1243.0,
    "processed_at": "2026-08-25T15:00:00Z"
  },
  "segments": [
    {
      "id": "seg_01",
      "start_s": 120.5,
      "end_s": 195.0,
      "ad_type": "midroll_sponsor_read",
      "confidence": 0.82,
      "brand": "Acme",
      "description": "Midroll Sponsor Read by Acme at 120.5s–195.0s (74.5s) (signals: sponsor_disclosure, discount_code)",
      "evidence": {
        "frame_timestamps": [121.0, 130.0, 150.0, 190.0],
        "transcript_span": "This episode is sponsored by Acme. Use code ACME20 ...",
        "signals_used": ["audio:sponsor_disclosure", "audio:discount_code", "ocr:ad_text_detected"]
      }
    }
  ],
  "stats": {
    "wall_clock_s": 87.3,
    "estimated_cost_usd": 0.0042,
    "frames_sampled": 45,
    "model_calls": 2
  }
}
```

---

## Web Viewer

After starting the API, open: http://localhost:8000/viewer

Features:
- Enter a YouTube URL or upload a local file
- Colored timeline bar showing detected ad segments by type
- Clickable segment cards (jump to timestamp)
- Confidence bar and signal tags per segment
- Processing stats panel

---

## Running the Test Set

```powershell
# Run all 5 assignment videos (skips Instagram Reels if local_path not set)
python scripts/run_test_set.py --output results/

# Results saved to results/yt_ujFWRFYLGjY.json etc.
# Combined file: results/all_results.json
```

### Instagram Reels

Download the two Instagram Reels manually, then update `test_data.json`:

```json
{
  "video_id": "ig_reel_DJl6-v8oufg",
  "local_path": "C:/path/to/reel_DJl6-v8oufg.mp4"
}
```

Then re-run `run_test_set.py`.

---

## Ground Truth Annotation

> **Important:** Do NOT annotate until you have watched each video.

```powershell
python -m adlens.evaluation.annotate --output ground_truth.json
```

The tool walks you through each test video interactively and saves results incrementally. See `EVAL.md` for the annotation methodology.

---

## Running Evaluation

After annotating and running the test set:

```powershell
python scripts/evaluate.py \
  --predictions results/all_results.json \
  --ground-truth ground_truth.json \
  --output eval_report.txt \
  --json-output eval_results.json
```

Outputs:
- `eval_report.txt` — human-readable report
- `eval_results.json` — machine-readable metrics

---

## Running Tests

```powershell
pytest tests/ -v

# Run specific test modules
pytest tests/test_evaluation.py -v
pytest tests/test_segmentation.py -v
pytest tests/test_asr.py -v
pytest tests/test_ocr.py -v
```

Tests do NOT require an API key — they test pure logic only.

---

## Instagram Reels

Instagram URLs are intentionally not scraped (anti-automation policy). The ingestion layer will raise a clear error if you pass an Instagram URL:

```
ValueError: Instagram URLs are not automatically scraped.
Please download the Reel manually and supply the local file path.
```

To analyze Instagram Reels:
1. Download the Reel manually (browser extension, screen recorder, etc.)
2. Use `POST /analyze/file` or update `test_data.json` with `local_path`

---

## Architecture

```
adlens/
├── api/                # FastAPI app + routes
│   ├── main.py         # App factory, lifespan, CORS
│   └── routes/
│       ├── analyze.py  # POST /analyze, POST /analyze/file
│       └── health.py   # GET /health
├── ingestion/          # Video ingesters (YouTube, local file)
├── audio/              # Audio extraction (ffmpeg wrapper)
├── asr/                # Whisper API, faster-whisper, mock + linguistic analysis
├── vision/             # Frame sampler, scene detector, visual classifier
├── ocr/                # EasyOCR, ad text filter
├── detection/          # Evidence aggregator, ad scorer (signal fusion)
├── segmentation/       # Segment builder, ad type classifier, Rule #8
├── evaluation/         # Metrics, matching, report generation
├── models/             # Shared Pydantic models
├── utils/              # Config, cost tracker, logger
├── web/                # Single-file HTML viewer
├── analyzer.py         # Main orchestrator (ties all stages together)
tests/                  # pytest unit tests
scripts/                # CLI scripts (run_test_set.py, evaluate.py)
main.py                 # uvicorn entry point
```

---

## Known Limitations

1. **No real ground truth yet** — Annotation tool is ready; results must be collected by watching the 5 test videos.
2. **Mock mode is not accurate** — Stub providers return deterministic fake results. Production mode requires API keys.
3. **EasyOCR downloads on first run** — ~300 MB of model weights. Subsequent runs use cache.
4. **faster-whisper downloads on first run** — ~150 MB for the small model.
5. **Long videos take time** — A 30-minute video at 1 FPS = 1800 frames. The 300-frame cap prevents excessive cost/time.
6. **Platform pre-rolls not detectable** — YouTube Ads inserted by the platform do not appear in downloaded files.
7. **Scene detection may fail on some codecs** — Falls back to uniform sampling silently.
8. **No authentication** — This is a prototype; do not expose publicly.
