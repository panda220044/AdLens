# AdLens

**Multimodal Video Advertisement Segment Detection System**

AdLens detects advertising segments in YouTube videos (VODs & Shorts) and local video files (including manually downloaded Instagram Reels) using a pipeline of Speech Recognition (ASR), OCR text extraction, visual classification, scene cut detection, and multimodal signal fusion. It returns a machine-readable timeline of every detected ad segment with timestamps, ad taxonomy type, confidence, brand, description, and supporting evidence.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Capabilities](#key-capabilities)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Application](#running-the-application)
9. [Instagram Reels Workflow](#instagram-reels-workflow)
10. [API Documentation](#api-documentation)
11. [Output Format](#output-format)
12. [Evaluation Framework](#evaluation-framework)
13. [Running Tests](#running-tests)
14. [Performance and Cost Instrumentation](#performance-and-cost-instrumentation)
15. [Known Limitations](#known-limitations)
16. [Reproducibility Guide](#reproducibility-guide)

---

## Project Overview

AdLens addresses the problem of identifying and categorizing commercial advertisements, sponsor reads, affiliate promotions, self-promotions, and bumpers in video streams. Rather than relying on a single modality or black-box model, AdLens processes video streams through 8 sequential, explainable pipeline stages:

```
Video URL / File → Ingestion → Audio Extraction & ASR → Scene Cut Detection 
                → Frame Sampling → OCR Text Extraction → Visual Classification 
                → Signal Fusion → Temporal Segmentation → Structured JSON Output
```

Each stage contributes modality-specific evidence (speech transcripts, OCR on-screen text, visual frame classifications, scene transition boundaries) that is fused into time windows and segmented into discrete ad intervals with full evidence provenance.

---

## Key Capabilities

* **Multimodal Detection**: Fuses audio transcript keywords/patterns, frame OCR text, visual image classification, and visual scene transition cuts.
* **YouTube & Local File Support**: Directly ingests YouTube VODs and YouTube Shorts URLs via `yt-dlp`, and processes local `.mp4`, `.mov`, `.mkv`, `.webm` files.
* **Flexible Provider Backends**:
  * **ASR**: Free Google SpeechRecognition, local `faster-whisper`, OpenAI Whisper API, or deterministic `mock`.
  * **Vision**: Local frame OCR heuristics (`local`), OpenAI `gpt-4o-mini` vision, or deterministic `mock`.
  * **OCR**: Local PyTorch deep learning OCR via `EasyOCR`.
  * **Scene Detection**: Histogram color transition detection via `PySceneDetect`.
* **Standardized Ad Taxonomy**: Categorizes segments into `preroll`, `midroll_sponsor_read`, `product_placement`, `self_promo`, `affiliate`, `bumper`, `platform_inserted`, or `other`.
* **Ambiguity Rulings & Rule #8 Merging**: Applies domain rules including Rule #3 (short bumper handling), Rule #7 (full video/reel ad detection), and Rule #8 (merging adjacent segments within 2s if they share the exact same brand).
* **Interactive Web Viewer**: Built-in dark-themed web interface for uploading files, pasting URLs, inspecting interactive timeline bars, and viewing confidence scores.
* **Cost & Performance Tracking**: Real-time instrumentation recording `wall_clock_s`, `estimated_cost_usd`, `frames_sampled`, and `model_calls`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AdLens Pipeline                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Video URL / Local Path                                  │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────┐                                    │
│  │  VideoIngester  │  yt-dlp (YouTube) / path (local)   │
│  └────────┬────────┘                                    │
│           │ local_path                                  │
│    ┌──────┴──────┬───────────┬──────────────┐           │
│    ▼             ▼           ▼              ▼           │
│  ┌──────┐  ┌──────────┐ ┌────────┐  ┌──────────────┐   │
│  │ ASR  │  │  Scene   │ │ Frame  │  │    Audio     │   │
│  │Whisp.│  │ Detector │ │Sampler │  │  Extractor   │   │
│  └──┬───┘  └────┬─────┘ └───┬────┘  └──────────────┘   │
│     │           │           │                           │
│     ▼           │           ▼                           │
│  ┌──────────┐   │      ┌─────────┐                      │
│  │Linguistic│   │      │   OCR   │  EasyOCR             │
│  │Analyzer  │   │      └────┬────┘                      │
│  └──┬───────┘   │           │                           │
│     │           │           ▼                           │
│     │           │      ┌────────────┐                   │
│     │           │      │  Visual    │  GPT-4o-mini /     │
│     │           │      │ Classifier │  Local Heuristic  │
│     │           │      └────┬───────┘                   │
│     │           │           │                           │
│     └───────────┴───────────┘                           │
│                     │                                   │
│                     ▼                                   │
│           ┌──────────────────┐                          │
│           │ Evidence         │  max-pool per window     │
│           │ Aggregator       │  (2s or 5s windows)      │
│           └────────┬─────────┘                          │
│                    │                                    │
│                    ▼                                    │
│           ┌──────────────────┐                          │
│           │   Ad Scorer      │  weighted fusion         │
│           │ (weighted sum)   │  audio+visual+ocr+scene  │
│           └────────┬─────────┘                          │
│                    │                                    │
│                    ▼                                    │
│           ┌──────────────────┐                          │
│           │ Segment Builder  │  threshold → group →     │
│           │                  │  refine → Rule #8        │
│           └────────┬─────────┘                          │
│                    │                                    │
│                    ▼                                    │
│           ┌──────────────────┐                          │
│           │  AnalysisResult  │  JSON API response       │
│           └──────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
d:\AdLens
├── .env                     # Environment variables (ignored by git)
├── .env.example             # Example environment configuration template
├── DESIGN.md                # System design & engineering rationale
├── EVAL.md                  # Evaluation methodology & metrics report
├── README.md                # Main documentation
├── pyproject.toml           # Pytest & project metadata
├── requirements.txt         # Dependency declarations
├── conftest.py              # Pytest configuration
├── main.py                  # Server entry point & FFmpeg PATH bootstrap
├── test_data.json           # Test set configuration (5 assignment videos)
├── adlens/                  # Core application package
│   ├── analyzer.py          # 8-stage pipeline orchestrator
│   ├── api/                 # FastAPI endpoints (/health, /analyze, /analyze/file)
│   ├── ingestion/           # YouTube and Local file ingesters
│   ├── audio/               # FFmpeg WAV audio extraction
│   ├── asr/                 # ASR providers & LinguisticAnalyzer
│   ├── vision/              # SceneDetector, FrameSampler, VisualClassifier
│   ├── ocr/                 # EasyOCRProvider & AdTextFilter
│   ├── detection/           # EvidenceAggregator & AdScorer
│   ├── segmentation/        # SegmentBuilder & taxonomy classifiers
│   ├── evaluation/          # Evaluation metrics & interactive annotator
│   ├── models/              # Pydantic schema primitives & API response models
│   ├── utils/               # Settings, CostTracker, logger
│   └── web/                 # Web viewer static assets (index.html)
├── scripts/                 # CLI scripts
│   ├── run_test_set.py      # Batch processor for test videos
│   └── evaluate.py          # Ground-truth evaluation report generator
├── tests/                   # Pytest unit tests (test_asr, test_ocr, etc.)
└── workdir/                 # Working directory for media files (ignored)
```

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on Python 3.10 – 3.14 |
| FFmpeg | Any recent | Required on system `PATH` for audio extraction and probing |
| OpenAI API Key | Optional | Required for `whisper_api` and `gpt4o_mini`. System runs 100% free locally without a key. |

### Installing FFmpeg (Windows)
```powershell
winget install Gyan.FFmpeg
```
Verify installation:
```powershell
ffmpeg -version
```

---

## Installation

```powershell
# 1. Enter the project directory
cd d:\AdLens

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Copy environment configuration template
Copy-Item .env.example .env
```

---

## Configuration

All configuration settings are managed via environment variables (or `.env` file). **No secrets or API keys are committed.**

### Key Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Optional OpenAI key for GPT-4o-mini and Whisper API. |
| `ADLENS_ASR_PROVIDER` | `free` | ASR provider: `free` (Google SpeechRec), `faster_whisper`, `whisper_api`, `mock`. |
| `ADLENS_VISION_PROVIDER` | `local` | Vision provider: `local` (OCR heuristics), `gpt4o_mini`, `mock`. |
| `ADLENS_SAMPLING_STRATEGY` | `adaptive` | Sampling strategy: `adaptive`, `uniform`, `keyframe`. |
| `ADLENS_SAMPLE_FPS` | `1.0` | FPS for uniform strategy. |
| `ADLENS_MAX_FRAMES_PER_VIDEO` | `300` | Hard limit cap on frame sampling (cost guard). |
| `ADLENS_AD_SCORE_THRESHOLD` | `0.40` | Minimum fused score threshold to flag an ad window. |
| `ADLENS_MAX_GAP_S` | `3.0` | Maximum gap tolerance for merging candidate segments. |
| `ADLENS_FUSION_AUDIO_WEIGHT` | `0.35` | Audio signal fusion weight. |
| `ADLENS_FUSION_VISUAL_WEIGHT` | `0.30` | Visual signal fusion weight. |
| `ADLENS_FUSION_OCR_WEIGHT` | `0.20` | OCR signal fusion weight. |
| `ADLENS_FUSION_SCENE_WEIGHT` | `0.15` | Scene cut signal fusion weight. |

---

## Running the Application

### 1. Starting the Backend API & Server
```powershell
python main.py
```
The server will start at `http://localhost:8000`.

### 2. Accessing the Interactive Web Viewer
Open your browser and navigate to:
```
http://localhost:8000/viewer
```
Features:
- Input a YouTube URL or upload a local video file.
- View detected ad segments rendered as a colored timeline bar.
- Interactive segment cards displaying timestamps, confidence, brand, and detected signals.
- Execution statistics panel (`wall_clock_s`, `estimated_cost_usd`, `frames_sampled`, `model_calls`).

### 3. Analyzing a YouTube Video (via API)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=ujFWRFYLGjY"}'
```

### 4. Analyzing a Local Video File (via API)
```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "video=@C:/path/to/local_video.mp4"
```

---

## Instagram Reels Workflow

Instagram enforces strict anti-bot and anti-automation protections on Reel URLs. To comply with ethical web practices:
* **AdLens intentionally does NOT scrape Instagram URLs directly.**
* Attempting to pass an Instagram URL (e.g. `https://www.instagram.com/reel/...`) to `/analyze` or `YouTubeIngester` will raise a `ValueError`:
  > *"Instagram URLs are not automatically scraped (anti-bot policy). Please download the Reel manually and supply the local file path."*

### Manual Workflow for Instagram Reels:
1. Download the Instagram Reel manually to your local disk using a web downloader or browser extension.
2. Analyze the downloaded Reel via the Web Viewer (`http://localhost:8000/viewer`) upload form, or via API endpoint `POST /analyze/file`.
3. To include the Reel in batch evaluation scripts (`scripts/run_test_set.py`), update `test_data.json` with the local path:
   ```json
   {
     "video_id": "ig_reel_DJl6-v8oufg",
     "local_path": "C:/path/to/reel_DJl6-v8oufg.mp4"
   }
   ```

---

## API Documentation

### Endpoints Summary

* `GET /health` — Service health status and provider configuration summary.
* `GET /viewer` — Serves the interactive HTML/JS Web Viewer interface.
* `POST /analyze` — Analyzes a video from a YouTube URL.
* `POST /analyze/file` — Analyzes a locally uploaded video file.

### Request / Response Examples

#### `GET /health`
**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "providers": {
    "asr": "free",
    "vision": "local",
    "sampling": "adaptive"
  },
  "openai_key_configured": false
}
```

#### `POST /analyze`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=ujFWRFYLGjY"
}
```

**Response:**
```json
{
  "source": {
    "url": "https://www.youtube.com/watch?v=ujFWRFYLGjY",
    "platform": "youtube",
    "kind": "vod",
    "duration_s": 1243.0,
    "processed_at": "2026-09-01T13:45:00Z"
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
        "transcript_span": "This episode is sponsored by Acme. Use code ACME20...",
        "signals_used": ["audio:sponsor_disclosure", "audio:discount_code", "ocr:ad_text_detected"]
      }
    }
  ],
  "stats": {
    "wall_clock_s": 45.2,
    "estimated_cost_usd": 0.0,
    "frames_sampled": 30,
    "model_calls": 0
  }
}
```

---

## Output Format

The output schema provides full structural evidence for downstream applications:
* `source`: Metadata including platform, video duration, and timestamp.
* `segments`: List of detected advertisement intervals:
  * `id`: Unique segment identifier (`seg_01`, `seg_02`).
  * `start_s` / `end_s`: Segment start and end timestamps in seconds.
  * `ad_type`: Taxon classification (`preroll`, `midroll_sponsor_read`, `product_placement`, `self_promo`, `affiliate`, `bumper`, `platform_inserted`, `other`).
  * `confidence`: Aggregated fused confidence score ($0.0 – 1.0$).
  * `brand`: Extracted brand name (or empty string if unspecified).
  * `description`: Human-readable summary description.
  * `evidence`: Frame timestamps, matched transcript text, and detected signal indicators.
* `stats`: Performance metrics (`wall_clock_s`, `estimated_cost_usd`, `frames_sampled`, `model_calls`).

---

## Evaluation Framework

AdLens includes an evaluation suite for measuring precision, recall, F1, IoU, and boundary error:

1. **Annotate Ground Truth**:
   ```powershell
   python -m adlens.evaluation.annotate --output ground_truth.json
   ```
   Interactive CLI tool to annotate ad boundaries for test videos. Note: The evaluation framework is fully implemented; actual ground truth numbers require completing manual annotation of test videos.

2. **Run Batch Test Set**:
   ```powershell
   python scripts/run_test_set.py --output results/
   ```
   Executes analysis across all 5 test videos and outputs individual and combined JSON results (`results/all_results.json`).

3. **Compute Metrics**:
   ```powershell
   python scripts/evaluate.py --predictions results/all_results.json --ground-truth ground_truth.json
   ```
   Generates `eval_report.txt` and `eval_results.json`.

---

## Running Tests

AdLens includes a suite of 48 unit tests covering ASR linguistic analysis, OCR filtering, signal fusion, segmentation, taxonomy classification, Rule #8 brand merging, and evaluation metrics.

To run the unit test suite:
```powershell
python -m pytest tests/ -v
```

### Verified Test Result
```
============================= 48 passed in 3.97s ==============================
```
* **Total Tests**: 48
* **Passed**: 48 (100% pass rate)
* **Failed**: 0

---

## Performance and Cost Instrumentation

Every analysis result automatically records 4 mandatory statistics:
* `wall_clock_s`: Total execution time in seconds from ingestion to segment building.
* `estimated_cost_usd`: Estimated API cost based on Whisper ($0.006/min) and GPT-4o-mini token pricing. Zero in local/free mode.
* `frames_sampled`: Total number of video frames extracted and processed.
* `model_calls`: Total number of external/local AI model inferences performed.

---

## Known Limitations

1. **Ground Truth Annotation**: Full metric calculation requires completing interactive ground truth annotation (`ground_truth.json`) by watching test videos.
2. **Platform Pre-Roll Ads**: Platform-inserted YouTube ads (TrueView/bumpers) are injected at player level and do not exist in downloaded `.mp4` video streams.
3. **Static Visual Sponsor Reads**: In host-delivered reads where visuals remain unchanged, detection relies primarily on ASR transcript signals.
4. **Local CPU OCR Latency**: EasyOCR initial run downloads PyTorch weights (~300MB) and runs on CPU if GPU is unavailable.

---

## Reproducibility Guide

To reproduce results from a clean environment:

1. Install Python 3.10+ and FFmpeg (`winget install Gyan.FFmpeg`).
2. Clone repository: `git clone https://github.com/panda220044/AdLens.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run test suite: `python -m pytest tests/ -v` (verify 48/48 pass).
5. Start server: `python main.py`
6. Open browser at `http://localhost:8000/viewer` and analyze videos.
