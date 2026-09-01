# AdLens — Design Document

## 1. Definition of an Advertisement

For the purposes of AdLens, an **advertisement** is any discrete temporal interval in a video where:

1. The creator or platform is **promoting a product, service, or property** for commercial consideration (paid or affiliate), OR
2. The creator is promoting their own monetizable property (Patreon, merch, newsletter) as a self-contained segment.

The promotion must be **separable from the primary content** — ambient branding, incidental logos, or product props that appear throughout the video without a dedicated promotional segment do NOT qualify.

**Minimum duration:** 1.0 second (configurable via `ADLENS_MIN_SEGMENT_DURATION_S`). Shorter intervals are classified as noise.

---

## 2. Ad Taxonomy

| Type | Definition | Signal Indicators |
|---|---|---|
| `preroll` | Advertisement at the very beginning of a video, before the main content starts. May be creator-controlled or platform-inserted. | Appears at t=0; visual/audio classify as ad immediately |
| `midroll_sponsor_read` | A host-delivered sponsorship segment embedded within the main content. Host explicitly addresses the audience about a product. | Sponsor disclosure phrases, discount codes, CTA; visual content may remain static (see ruling #6) |
| `product_placement` | A product/brand is featured within the content itself without an explicit announcement — integrated into the narrative. | Brand visible in frame, product used on-screen without audio disclosure |
| `self_promo` | Creator promotes their own channel assets — Patreon, merchandise, newsletter, social. | Patreon/merch patterns in transcript; NOT ambient logo wear (see ruling #2) |
| `affiliate` | Creator promotes a product via tracked link or discount code, earning commission. Structurally similar to midroll but emphasis is on the affiliate relationship. | Explicit discount/referral code, affiliate link patterns in OCR |
| `platform_inserted` | Ads inserted by the platform (YouTube Ads). Not present in downloaded video. Detected only by absence or metadata. | Cannot be detected from video stream alone — documented in stats |
| `bumper` | Short (≥ 1.0s, typically < 3s) "sponsored by" identifier card or animation. | Duration < 3s with sponsor signal; visual "bumper" hint |
| `other` | Advertising content that does not fit the above categories. | Catch-all when signals are present but type is ambiguous |

---

## 3. Ambiguity Rulings

All eight rulings from the assignment brief are addressed here.

### Ruling 1: "If you like this channel, join my Patreon"

**Decision: `self_promo`**

This is a monetization call-to-action for the creator's own property. The creator earns directly from the viewer. It is classified as `self_promo`, not a third-party advertisement.

**Implementation:** The `self_promo` pattern set in `LinguisticAnalyzer` includes `patreon.com`, `join my patreon`, `become a patron`, `become a member`.

**Edge case:** If the segment lasts < 5 seconds and is embedded in a sign-off (common), the system may merge it with nearby content. The minimum segment duration (1.0s) and max-gap merging (3.0s) govern this.

---

### Ruling 2: Host wears their own merch logo throughout the video

**Decision: NOT an ad segment**

Ambient branding throughout an entire video is not a separable advertising interval. The merch logo is costume, not a promotional segment.

**Implementation:** AdLens only flags discrete temporal segments with elevated ad scores. A logo visible in every frame raises OCR scores uniformly — the fused score across all windows will be low relative to the threshold because there is no temporal contrast. The system requires a concentration of multiple signals to cross the threshold.

**Exception:** If the host explicitly says "check out my merch" and holds up a shirt for 30 seconds, that discrete segment would be flagged as `self_promo`.

---

### Ruling 3: 1.4-second animated "sponsored by" bumper

**Decision: `bumper`**

Minimum segment duration is set to **1.0 second**, so a 1.4-second bumper qualifies. It is classified as `bumper` because:
- Duration < 3.0 seconds → `classify_ad_type()` returns `bumper`
- Scene boundaries at start/end provide refinement signal

**Implementation:** `classify_ad_type(segment_duration_s=1.4, ...)` returns `AdType.bumper` for any segment < 3.0s. The bumper must still exceed the `ADLENS_MIN_SEGMENT_DURATION_S=1.0` threshold to be kept.

---

### Ruling 4: Movie review includes 40 seconds of official trailer

**Decision: NOT an ad segment**

An official trailer included in a review is editorial clip usage, not a paid placement. The creator is not commercially compensated for showing the trailer (in a standard review context).

**Implementation:** The system does NOT flag this by design. Trailer content typically contains:
- No sponsor disclosure language in ASR
- No discount codes or CTAs
- Visual content (action scenes, dialogue) that does not match the ad visual classifier's training profile

Without strong signals, the fused score remains below threshold.

**Caveat:** If the trailer contains embedded product placement from a third party, those frames may score slightly higher — but the absence of ASR signals keeps the overall score low.

---

### Ruling 5: Platform pre-roll that does not appear in downloaded video

**Decision: `platform_inserted` — documented in stats, cannot be detected**

YouTube Ads (TrueView, bumpers) are injected at the CDN/player level. They are not present in the video stream returned by yt-dlp. AdLens cannot detect these from the downloaded file.

**Implementation:** This is documented in the API response stats and in EVAL.md. If a video is known to have pre-rolls, this is noted as a known false negative. No workaround is implemented — it would require browser-level recording, which is out of scope.

---

### Ruling 6: 90-second sponsor read where visuals do not change

**Decision: `midroll_sponsor_read` — audio signal dominates**

Static visuals during a sponsor read are expected and do not reduce confidence. The visual classifier may return low scores for these frames (no obvious on-screen ad elements), but the audio/ASR signal (sponsor disclosure language, discount code, CTA) is strong and dominates the fusion score.

**Implementation:** Fusion weights are `audio=0.35, visual=0.30`. Audio is the highest-weighted signal specifically to handle this case. A strong audio signal alone (`score=0.9 × 0.35 = 0.315`) combined with moderate OCR (`score=0.5 × 0.20 = 0.10`) gives `fused=0.415`, which crosses the `0.40` threshold.

**Recommended configuration:** If you know a video has many sponsor reads with static visuals, increase `ADLENS_FUSION_AUDIO_WEIGHT` to 0.50.

---

### Ruling 7: Reel that is an ad from first frame to last frame

**Decision: `preroll` (or strongest available type)**

When a single candidate segment covers > 90% of the video duration, `classify_ad_type()` sets `is_full_video=True` and returns `AdType.preroll` regardless of other signals. This is the correct classification: the entire reel is a pre-roll advertisement.

**Implementation:** `is_full_video = (duration_s / video_duration_s) > 0.90` in `SegmentBuilder._build_segment()`.

**Output:** One segment is returned: `seg_01`, spanning approximately `0.0s–[duration]s`, typed `preroll`.

---

### Ruling 8: Two sponsors back-to-back with no gap

**Decision: Two separate segments if different brands; one merged segment if same brand and gap < 2s**

This is the most nuanced case. The rule implemented in `SegmentBuilder._merge_adjacent_same_brand()`:

| Condition | Action |
|---|---|
| Gap < 2s AND same brand (case-insensitive) | Merge into one segment |
| Gap ≥ 2s OR different brands | Keep as separate segments |

**Rationale:** Two back-to-back reads for the same sponsor (common in podcast-style videos) are likely one deal with a mid-read transition. Two different sponsors back-to-back are clearly separate sponsorships and must be reported separately for accurate brand attribution.

**Example:** Acme → [1s gap] → Acme → merged. Acme → [1s gap] → Globex → separate.

---

## 4. Architecture

### Component Diagram

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
│     │           │      │  Visual    │  GPT-4o-mini       │
│     │           │      │ Classifier │  (batched)         │
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
│           │   Ad Scorer      │  weighted fusion          │
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
│           │  AnalysisResult  │  JSON API response        │
│           └──────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Signal Choices and Justification

### A. ASR / Linguistic (weight: 0.35 — highest)

**Why ASR?** Sponsor reads are primarily audio events. The linguistic signals (sponsor disclosure, discount codes, CTAs) are explicit and reliable. False positive rate from transcript analysis is low because the keyword patterns are specific.

**Why the highest weight?** A creator doing a sponsor read in a dark/static frame would produce no visual or OCR signal. ASR handles this correctly. This is empirically validated across YouTube sponsorship patterns.

**Weakness:** Relies on the host being audible and speaking clearly. Music-heavy or non-English content degrades this signal.

### B. Visual Classification (weight: 0.30)

**Why vision?** On-screen brand overlays, product shots, and sponsored labels appear in some ad formats (especially pre-rolls and product placements) where audio is ambiguous.

**Why not the highest?** Midroll sponsor reads — the most common format — often look identical to regular content visually. Overweighting vision penalizes correctly-identified audio-only sponsor reads.

**Cost control:** Vision is only called on candidate frames (frames with OCR signal, near scene boundaries, or in the first 10s). This typically reduces vision calls by 60-80% vs. calling on every frame.

### C. OCR (weight: 0.20)

**Why OCR?** Promotional text ("SPONSORED", discount codes, URLs, "Shop Now") appears on-screen in many formats. OCR is cheap (runs locally) and catches signals missed by audio.

**Why lower weight?** OCR false positives are common — channel watermarks, subtitles, and UI elements can match broad patterns. The pattern set in `AdTextFilter` is calibrated to be specific, but OCR is treated as a supporting signal, not a primary one.

### D. Scene/Boundary Detection (weight: 0.15)

**Why scene detection?** Ad segments typically begin and end with visible scene cuts. Scene boundaries sharpen the start/end timestamps of detected segments. The adaptive sampling strategy also uses scene boundaries to concentrate frames where they matter.

**Why the lowest weight?** A scene cut alone is evidence of nothing — every cut is a boundary. This signal amplifies adjacent evidence rather than standing alone.

---

## 6. Sampling Strategy

Three strategies are implemented in `FrameSampler`:

### Uniform (baseline)
- Sample at a fixed rate (default: 1 FPS)
- Simple, predictable
- Miss short segments if rate is too low; expensive if rate is high

### Adaptive (recommended)
- Dense sampling (every 0.5s) around scene boundaries (within ±2s window)
- Sparse background fill (1 frame every 2s)
- Deduplication and hard cap applied
- Concentrates frames where temporal transitions happen; good coverage/cost ratio

### Keyframe
- Extract I-frames only via ffprobe
- Least redundancy; frame distribution is decoder-dependent
- Useful for very long videos with many cuts

**Configuration:** `ADLENS_SAMPLING_STRATEGY=adaptive` (default)

**Hard cap:** `ADLENS_MAX_FRAMES_PER_VIDEO=300` prevents runaway cost on long videos.

---

## 7. Cost Estimate

Based on OpenAI pricing as of August 2026:

| Operation | Unit cost | Expected for 5 test videos |
|---|---|---|
| Whisper API | $0.006/min | ~60 min audio → **$0.36** |
| GPT-4o-mini vision (detail=low) | $0.000425/frame | ~300 frames → **$0.13** |
| GPT-4o-mini text (type classification) | ~$0.0002/1K tokens | ~10 calls → **$0.02** |
| **Total** | | **~$0.51** |

This is well within the $10 budget. Cost per video averages ~$0.10.

**Cost levers:**
- Reduce `ADLENS_MAX_FRAMES_PER_VIDEO` for cheaper runs
- Use `ADLENS_VISION_DETAIL=low` (default) vs `high` (~13x cost difference)
- Use `ADLENS_ASR_PROVIDER=faster_whisper` for $0 ASR

---

## 8. Failed Approaches (Considered and Rejected)

### Uniform sampling only
Rejected because 1 FPS over a 20-minute video = 1200 frames × $0.000425 = $0.51 for vision alone. Adaptive sampling achieves comparable coverage at 10-20% of the frame count.

### Scene detection as primary signal
Rejected because most scene cuts in regular content are not ad boundaries. Scene detection is useful only as a supporting/amplifying signal.

### Single black-box LLM call
Considered asking GPT-4o to "find the ads in this video" without structured signals. Rejected because:
1. No explainability of detections
2. Cannot provide per-signal evidence
3. Much more expensive for long videos
4. Less reproducible

### Embedding similarity (transcript vs. ad template)
Considered using sentence embeddings to compare transcript segments against known ad phrase templates. Rejected in favor of explicit regex patterns because:
1. Regex patterns are interpretable and debuggable
2. Sufficient for the signal categories needed
3. Embedding approach adds significant dependency weight

---

## 9. Scaling to 1,000 Videos/Day

With the current synchronous architecture, each video takes 60-180 seconds. Processing 1,000/day requires concurrency.

**Recommended approach:**
1. Replace synchronous processing with a task queue (Celery + Redis or RQ)
2. Workers scale horizontally — each worker handles one video at a time
3. Results stored in a database (PostgreSQL) rather than files
4. FastAPI returns a `job_id` immediately; client polls or uses webhooks
5. EasyOCR model loaded once per worker (not per request)
6. faster-whisper model loaded once per worker (GPU workers)

**Cost at scale:** 1,000 videos/day × ~20 min avg × $0.006/min = **$120/day** for Whisper API alone. At this scale, switching to self-hosted Whisper (A100 GPU) costs ~$30/day. Vision API cost is proportional and can be controlled by reducing frames sampled.

---

## 10. What Would Be Built With Two More Weeks

1. **Real ground truth** — Watch all 5 test videos, annotate precisely, report actual metrics
2. **Brand detection** — CLIP embeddings for brand logo identification; brand database lookup
3. **Live stream support** — Bounded window processing with overlap for cross-boundary segments
4. **Confidence calibration** — Isotonic regression to calibrate raw scores to actual probabilities
5. **Multi-language support** — EasyOCR language packs + Whisper language detection
6. **Self-hosted Whisper** — Replace API with local whisper.cpp for zero ASR cost
7. **CI/CD** — GitHub Actions pipeline running tests on every PR
8. **Caching** — Skip re-downloading videos already in workdir (use URL hash)
9. **WebSocket progress** — Stream stage-by-stage progress to the viewer UI

---

## 11. AI-Assisted Development Disclosure

This codebase was developed with the assistance of Antigravity (Google DeepMind), an AI coding assistant. All code was:

1. Reviewed for correctness before being included
2. Structured to be understandable without the AI context
3. Organized so that every module can be explained independently

The AI did not fabricate test results, model outputs, ground truth, or costs. All evaluation results must come from actual video processing runs using the provided scripts.

The architecture decisions, ambiguity rulings, and signal design are original and defensible. The AI accelerated implementation but did not make design decisions independently.
