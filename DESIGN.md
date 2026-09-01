# AdLens — System Design & Engineering Specification

## 1. Problem Definition

In modern digital video content across platforms like YouTube and Instagram, commercial advertisements take diverse forms:
- **Platform-inserted ads**: Dynamic video ads injected by hosting platforms.
- **Embedded sponsor reads**: Creator-delivered promotional messages recorded directly inside the video stream.
- **Product placements**: Visual showcase of brand products or logos within content.
- **Self-promotions**: Creator calls-to-action for their own monetizable assets (Patreon, merchandise, newsletters).
- **Affiliate promotions**: Discount codes and links yielding sales commissions.
- **Short bumpers**: Brief 1-2 second animated "sponsored by" intro/outro cards.

Detecting these segments automatically and extracting accurate time boundaries, ad taxonomy categories, brands, and confidence levels is challenging because no single modality is sufficient:
- A host may talk about a sponsor while visuals remain completely static.
- A product placement may appear on screen silently with no speech disclosure.
- An on-screen banner may show a discount code while the creator speaks about an unrelated topic.

**AdLens** addresses this challenge through a multi-modal signal fusion architecture that combines Automatic Speech Recognition (ASR), Optical Character Recognition (OCR), Computer Vision classification, and Visual Scene Cut Detection into an aggregated probability stream, which is then segmented into discrete ad intervals with full evidence provenance.

---

## 2. What Counts as an Advertisement

For the purposes of AdLens, an **advertisement** is defined as any discrete temporal interval within a video where:
1. The creator or hosting platform promotes a commercial product, service, or brand for financial consideration or commission (e.g. paid sponsorship, affiliate code).
2. The creator explicitly promotes their own monetizable assets (Patreon, merchandise store, paid newsletter, channel memberships) as a dedicated segment.

### Inclusion Criteria
* Host-delivered sponsorship reads ("This episode is brought to you by...").
* Display of affiliate discount codes, custom referral URLs, or promotional lower-thirds.
* On-screen banner overlays or animated "Sponsored by" bumpers.
* Dedicated creator self-promotion segments for paid memberships or merch.

### Exclusion Criteria
* **Ambient logo wear**: Creators wearing clothing with brand logos throughout the video without a dedicated promotional read.
* **Incidental background props**: General office hardware, laptops, or drinks appearing naturally in shot without active promotion.
* **Editorial clip usage**: Reviewing or analyzing movie trailers, games, or third-party clips in an educational or critical context.
* **General channel subscriptions**: Standard unpaid end-screen reminders ("Like and subscribe for more free videos").

---

## 3. Advertisement Taxonomy

AdLens categorizes detected segments into eight standardized taxonomy types:

| Taxonomy Type | Definition | Key Signal Indicators |
|---|---|---|
| `preroll` | Advertisement playing before main content begins (or entire video reel when 100% ad). | Timestamp $t=0$; high fused score; duration $>90\%$ of total reel length. |
| `midroll_sponsor_read` | Creator-delivered sponsorship read integrated within main content. | Sponsor disclosure phrases ("sponsored by"), affiliate codes, CTAs; static visuals allowed. |
| `product_placement` | On-screen showcase of a product/brand integrated into narrative without audio disclosure. | Visual brand detection, OCR brand text, absence of speech disclosure. |
| `self_promo` | Promotion of creator-owned monetizable property (Patreon, merch, newsletter). | ASR keywords ("Patreon", "my merch", "newsletter"), OCR links. |
| `affiliate` | Promotion emphasizing a trackable discount code or referral link for commission. | Specific promo codes ("use code X"), percentage off phrases ("20% off"). |
| `platform_inserted` | Platform-level injected ads (e.g. YouTube TrueView). Not present in downloaded files. | Tracked in stats metadata only; non-extractable from video stream. |
| `bumper` | Short ($\ge 1.0\text{s}$, $< 3.0\text{s}$) animated or static "sponsored by" card. | Segment duration $<3.0\text{s}$; visual "bumper" hint or scene cut boundary. |
| `other` | Commercial promotional content not fitting neatly into specific categories. | Catch-all when fused score exceeds threshold without specific signal match. |

---

## 4. Ambiguity Rulings

AdLens enforces explicit, deterministic rulings for common edge cases:

### Ruling 1: "If you like this channel, join my Patreon"
* **Decision**: `self_promo`
* **Rationale**: Direct monetization call-to-action for creator-owned property.
* **Implementation**: `LinguisticAnalyzer` matches patterns like `join my patreon`, `become a patron`, `merch store`.

### Ruling 2: Host wears their own merch logo throughout the video
* **Decision**: **NOT an ad segment**
* **Rationale**: Ambient branding lacks temporal boundaries and temporal contrast.
* **Implementation**: Max-pooling across time windows requires temporal signal contrast; static background logos produce low relative fused scores below threshold.

### Ruling 3: 1.4-second animated "sponsored by" bumper
* **Decision**: `bumper`
* **Rationale**: Bumper cards are brief commercial identifiers.
* **Implementation**: Minimum duration threshold is set to $1.0\text{s}$. Segments with duration $<3.0\text{s}$ are classified as `bumper`.

### Ruling 4: Movie review includes 40 seconds of official trailer
* **Decision**: **NOT an ad segment**
* **Rationale**: Editorial fair-use clip inclusion, not commercial sponsorship.
* **Implementation**: Lack of sponsor disclosures, promo codes, or affiliate CTAs keeps speech and OCR scores at zero.

### Ruling 5: Platform pre-roll not present in downloaded file
* **Decision**: `platform_inserted` (Documented limitation)
* **Rationale**: Stream download tools (`yt-dlp`) extract content streams after platform ad injection.
* **Implementation**: Logged in metadata stats. Cannot be detected from stream bytes alone.

### Ruling 6: 90-second sponsor read where visuals do not change
* **Decision**: `midroll_sponsor_read`
* **Rationale**: Audio speech signals are primary for host sponsor reads.
* **Implementation**: Audio signal weight ($0.35$) dominates fusion calculation; strong transcript disclosure crosses the $0.40$ threshold even if visual score is low.

### Ruling 7: Short Reel that is an ad from first frame to last frame
* **Decision**: `preroll`
* **Rationale**: When candidate segment duration spans $>90\%$ of total video length, the entire video is a promotional ad.
* **Implementation**: `is_full_video = (duration / video_duration) > 0.90` sets taxonomy to `preroll`.

### Ruling 8: Two sponsors back-to-back with no gap
* **Decision**: Merge if same brand and gap $<2.0\text{s}$; keep separate if different brands.
* **Rationale**: Same-brand back-to-back reads represent one continuous placement; different brands represent separate commercial deals.
* **Implementation**: `SegmentBuilder._merge_adjacent_same_brand()` checks brand equality and temporal gap.

---

## 5. Detection Signals & Modality Rationale

AdLens leverages four complementary detection signals:

1. **Audio Speech / ASR Transcripts (`weight = 0.35`)**:
   * *Rationale*: Speech transcript disclosures ("sponsored by", "use code", "link in description") are the most reliable indicator of sponsor reads.
2. **Visual Classification (`weight = 0.30`)**:
   * *Rationale*: Detects graphical ad layouts, product hero shots, lower-third overlays, and brand graphics.
3. **OCR On-Screen Text (`weight = 0.20`)**:
   * *Rationale*: Captures visual discount codes, website URLs, "SPONSORED" labels, and price points.
4. **Scene Cut Boundaries (`weight = 0.15`)**:
   * *Rationale*: Visual scene cuts mark transition boundaries between organic content and sponsor segments.

---

## 6. Signal Fusion & Scoring Logic

The video stream is divided into non-overlapping temporal windows ($2.0\text{s}$ for short videos $<120\text{s}$, $5.0\text{s}$ for long-form videos).

### Window Scoring Formula
For each window $W_i$, individual modality scores $S_{\text{audio}}, S_{\text{visual}}, S_{\text{ocr}}, S_{\text{scene}} \in [0.0, 1.0]$ are aggregated via max-pooling within the window time range.

$$\text{Base Score} = w_{\text{audio}} \cdot S_{\text{audio}} + w_{\text{visual}} \cdot S_{\text{visual}} + w_{\text{ocr}} \cdot S_{\text{ocr}}$$

To ensure visual-only advertisements (e.g. poster overlays or banner ads without speech) are detected:
$$\text{If } \max(S_{\text{visual}}, S_{\text{ocr}}) \ge 0.50 \implies \text{Base Score} = \max\left(\text{Base Score}, \max(S_{\text{visual}}, S_{\text{ocr}}) \cdot 0.85\right)$$

Scene cut transitions amplify nearby evidence:
$$\text{Scene Boost} = w_{\text{scene}} \cdot S_{\text{scene}} \cdot \begin{cases} 1.0 & \text{if Base Score} > 0.10 \\ 0.20 & \text{otherwise} \end{cases}$$

$$\text{Fused Score}(W_i) = \min\left(\text{Base Score} + \text{Scene Boost}, 1.0\right)$$

Candidate windows are selected where $\text{Fused Score}(W_i) \ge 0.40$.

---

## 7. Temporal Segmentation & Boundary Logic

1. **Candidate Grouping**: Adjacent candidate windows separated by less than `max_gap_s` ($3.0\text{s}$ default) are merged into candidate temporal intervals.
2. **Duration Filtering**: Candidates shorter than `min_segment_duration_s` ($1.0\text{s}$) are filtered out as noise.
3. **Boundary Refinement**: Segment start and end timestamps ($\text{start\_s}, \text{end\_s}$) are aligned with adjacent scene boundaries if a scene cut occurs within $2.0\text{s}$ of the window boundary.
4. **Taxonomy & Brand Classification**: `classify_ad_type()` assigns the ad type, and `extract_brand()` extracts brand entities from visual and OCR evidence.
5. **Rule #8 Same-Brand Merging**: Adjacent segments separated by $<2.0\text{s}$ with matching brand names are merged into a single segment.

---

## 8. Short Advertisement Handling

Short commercial bumpers ($1.0\text{s} - 3.0\text{s}$) present unique challenges due to low frame counts:
- Adaptive sampling extracts extra frames at scene cut points.
- Any candidate segment $<3.0\text{s}$ duration passing the $1.0\text{s}$ minimum threshold is classified as `bumper`.

---

## 9. False Positive / False Negative Considerations

- **False Positive Controls**:
  - Requires multi-signal fusion or strong individual OCR/Visual score ($\ge 0.50$).
  - Ignores ambient static text through window max-pooling contrast.
  - Excludes unpaid channel subscription reminders ("subscribe for more videos").
- **False Negative Controls**:
  - High audio weight ($0.35$) catches sponsor reads with static visuals.
  - Visual/OCR score boosting catches silent banner overlays.
  - Gap tolerance ($3.0\text{s}$) bridges brief pauses in speech during host reads.

---

## 10. Model and Tool Choices

- **Video Ingestion**: `yt-dlp` (high reliability for YouTube format extraction).
- **Audio Processing**: `FFmpeg` (fast 16kHz mono WAV extraction).
- **ASR**: `SpeechRecognition` (Google free API) for zero-cost operation; `faster-whisper` for offline local inference; OpenAI `whisper-1` for cloud API accuracy.
- **Scene Cut Detection**: `PySceneDetect` (`ContentDetector` HSV color space analysis).
- **Frame Sampling**: OpenCV (`cv2`) for frame decoding.
- **OCR**: `EasyOCR` (local PyTorch deep learning OCR model).
- **Visual Classification**: `LocalVisualClassifier` (OCR heuristics) for zero-cost offline runs; OpenAI `gpt-4o-mini` for multimodal LLM vision.
- **API Framework**: `FastAPI` + `Uvicorn` for async web serving.

---

## 11. Cost and Latency Considerations

- **Cost Safety**: `ADLENS_MAX_FRAMES_PER_VIDEO` (default 300) hard-caps vision API calls.
- **Batched Vision Requests**: Frames sent to GPT-4o-mini in batches of 5 with `detail="low"` (~$0.0004/frame).
- **Candidate Pre-filtering**: Vision LLM classification is executed only on frames exhibiting prior OCR or scene signals.
- **Local Fallback**: Zero API key requirement when running in `free`/`local` mode.

---

## 12. Reliability and Failure Modes

- **Graceful Modality Degradation**: If ASR, OCR, or scene detection fails on a video stream, the error is logged and the pipeline continues with remaining available modalities.
- **FFmpeg PATH Bootstrapping**: `main.py` automatically inspects standard WinGet, Scoop, and system directories to locate `ffmpeg.exe` if not present in process PATH.
- **Deterministic Stubs**: `mock` providers allow offline integration testing without network or GPU dependencies.

---

## 13. System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                              AdLens API                                │
│                   (FastAPI / Uvicorn @ port 8000)                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           AdLensAnalyzer                               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Ingest (YouTube / Local File)                                       │
│ 2. Extract Audio (FFmpeg) & Transcribe ASR (Whisper / Free)            │
│ 3. Linguistic Analysis (Keywords / Patterns)                           │
│ 4. Detect Scene Cuts (PySceneDetect)                                   │
│ 5. Sample Frames (Adaptive Scene-Aware)                                │
│ 6. Extract On-Screen Text (EasyOCR) & Filter Ad Text                  │
│ 7. Classify Frames (GPT-4o-mini / Local Visual)                        │
│ 8. Aggregate & Score Windows (Signal Fusion)                           │
│ 9. Build Segments (Rule #3, Rule #7, Rule #8 Merging)                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AnalysisResult (JSON)                           │
│  - source: metadata                                                    │
│  - segments: [AdSegment(start_s, end_s, ad_type, confidence, brand)]  │
│  - stats: ProcessingStats(wall_clock_s, cost, frames, model_calls)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Design Trade-offs

| Decision | Advantage | Trade-off |
|---|---|---|
| Fixed-width time windows (2s/5s) | Fast, deterministic max-pooling signal fusion. | Boundary precision is capped by window resolution before refinement. |
| Heuristic OCR filtering before LLM vision | Reduces LLM API calls by $>70\%$. | May skip vision LLM on subtle visual placements lacking text. |
| Local EasyOCR vs Cloud Vision | $100\%$ free, zero API cost. | PyTorch CPU execution adds initial model load latency. |

---

## 15. Reproducibility

The system is designed for $100\%$ reproducible execution across environments:
- Zero hardcoded environment paths or secrets.
- Environment variables configured via `.env`.
- Unit test suite (`python -m pytest tests/ -v`) verifies pipeline logic deterministically without API keys.

---

## 16. Known Limitations & Future Improvements

### Limitations
1. **Platform Pre-Roll Ads**: Cannot detect platform-injected ads removed during video stream downloading.
2. **Manual Ground Truth Annotation**: Evaluation metrics code is fully implemented; actual dataset metrics require completing interactive manual annotation.

### Future Improvements
1. **Audio Music/Jingle Detection**: Add acoustic classifier for distinct sponsor intro music.
2. **Native GPU Acceleration**: Auto-detect CUDA for faster EasyOCR and local Whisper inference.
3. **Live Stream Chunking**: Extend pipeline to process live HLS video streams in real-time.
