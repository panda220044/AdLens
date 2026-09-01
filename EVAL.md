# AdLens — Evaluation Document

## 1. Ground Truth Methodology

### Principles

1. **No fabrication** — Ground truth labels are only recorded after actually watching each video.
2. **Conservative annotation** — When in doubt, do NOT annotate a segment as an ad. False negatives are preferable to false positives in the ground truth.
3. **Temporal precision** — Start and end times are recorded to the nearest 0.5 second.
4. **Independent annotation** — A single annotator reviews each video. For production evaluation, two annotators with adjudication would be standard.

### Annotation Process

1. Watch the full video in a browser or media player with timestamps visible
2. Note the start/end time of every segment that meets the ad definition (see DESIGN.md §1)
3. Classify each segment using the taxonomy (see DESIGN.md §2)
4. Record brand if identifiable
5. Add notes explaining the ruling for ambiguous cases

### Tooling

```bash
python -m adlens.evaluation.annotate --output ground_truth.json
```

The tool prompts for each test video interactively and saves results incrementally after each video.

---

## 2. Ground Truth Status

> [!IMPORTANT]
> **Ground truth has NOT yet been collected.**
>
> The annotation tool is implemented and ready. The 5 test videos must be watched and annotated before evaluation metrics can be reported.
>
> The table below will be filled in after annotation is complete.

| Video ID | URL | Platform | Annotated | Segments Found |
|---|---|---|---|---|
| `yt_ujFWRFYLGjY` | youtube.com/watch?v=ujFWRFYLGjY | YouTube VOD | ❌ | — |
| `yt_shorts_Ve0zdhTQA4U` | youtube.com/shorts/Ve0zdhTQA4U | YouTube Short | ❌ | — |
| `yt_s0LLVQeMmtU` | youtube.com/watch?v=s0LLVQeMmtU | YouTube VOD | ❌ | — |
| `ig_reel_DJl6-v8oufg` | instagram.com/reel/DJl6-v8oufg | Instagram Reel | ❌ | — |
| `ig_reel_Db78RoIuOyl` | instagram.com/reel/Db78RoIuOyl | Instagram Reel | ❌ | — |

---

## 3. Metric Definitions

### Segment Detection Metrics

Evaluation is performed at the **segment level**, not at the frame level. Two segments are considered a match if their IoU ≥ 0.50.

#### Precision
$$P = \frac{TP}{TP + FP}$$

Of all predicted segments, what fraction are correct (have a matching ground truth segment with IoU ≥ 0.50)?

#### Recall
$$R = \frac{TP}{TP + FN}$$

Of all ground truth segments, what fraction did the system find?

#### F1
$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

Harmonic mean of precision and recall. Balances the two; primary metric for comparison.

#### Segment IoU
$$\text{IoU}(a, b) = \frac{|a \cap b|}{|a \cup b|}$$

For segments $a = [a_{\text{start}}, a_{\text{end}}]$ and $b = [b_{\text{start}}, b_{\text{end}}]$:

$$\text{IoU} = \frac{\max(0, \min(a_{\text{end}}, b_{\text{end}}) - \max(a_{\text{start}}, b_{\text{start}}))}{\max(a_{\text{end}}, b_{\text{end}}) - \min(a_{\text{start}}, b_{\text{start}})}$$

Mean IoU is computed over matched pairs only.

#### Boundary Error
- **Start error** = |predicted_start − gt_start|, in seconds
- **End error** = |predicted_end − gt_end|, in seconds
- Computed as RMSE over all matched pairs

### Macro vs Micro Aggregation

- **Macro**: Average of per-video metrics. Treats each video equally regardless of number of segments.
- **Micro**: Computed from aggregate TP/FP/FN counts. Dominated by videos with more segments.

Both are reported. For the 5-video test set they will differ only if videos have very different numbers of segments.

---

## 4. Per-Video Results

> [!NOTE]
> **Results will be populated after running the analysis pipeline on real videos with production providers.**
>
> Running command:
> ```bash
> python scripts/run_test_set.py --output results/
> python scripts/evaluate.py --predictions results/all_results.json --ground-truth ground_truth.json
> ```

### Mock Mode Results (Development Only)

The following results are from mock provider mode and are NOT meaningful evaluation results. They demonstrate that the pipeline runs end-to-end.

| Video ID | Segments Predicted | Notes |
|---|---|---|
| `yt_ujFWRFYLGjY` | 1 (mock) | Mock ASR injects fake sponsor at 30-90s |
| `yt_shorts_Ve0zdhTQA4U` | 0 or 1 | Short video — mock may not trigger |
| `yt_s0LLVQeMmtU` | 1 (mock) | Same as above |
| Instagram Reels | Skipped | local_path not configured |

**These are not evaluation results. Do not treat them as such.**

---

## 5. Failure Cases

The following failure categories are anticipated based on system design analysis. Actual failure analysis will be completed after real runs.

### Anticipated False Negatives (Missed Ads)

| Case | Cause | Mitigation |
|---|---|---|
| Very quiet sponsor read | Host speaks quietly; ASR has low confidence | Use larger Whisper model (medium vs. small) |
| Purely visual product placement | No audio signal; product appears on-screen silently | Vision signal must carry it; requires real GPT-4o-mini |
| Platform-inserted pre-roll | Not in downloaded video | Document as known limitation |
| Non-English sponsor disclosure | Linguistic patterns are English-only | Add per-language keyword sets |

### Anticipated False Positives (Incorrect Detections)

| Case | Cause | Mitigation |
|---|---|---|
| Movie/TV clip in review | Dialogue may match CTA patterns | Lower fusion weights or add negation heuristic |
| Host discusses ad-related topics | "I don't use referral codes" triggers patterns | Add negation detection to linguistic analyzer |
| Channel watermarks | OCR detects channel logo as "brand" | Filter known channel-specific text patterns |
| Dense scene cutting (action video) | Many scene boundaries → elevated scene score | Ensure scene signal alone cannot cross threshold |

---

## 6. Root-Cause Analysis Framework

For each false negative or false positive encountered during real evaluation:

1. **Which signals fired?** Check `evidence.signals_used` in the API response
2. **What was the fused score?** Compare against threshold
3. **Which signal was absent?** Identify the missing modality
4. **Configuration fix?** Adjust weights, threshold, or provider
5. **Code fix?** Add pattern, fix regex, update OCR filter

Failure analysis will be documented as: `[FP|FN]-[video_id]-[segment_id]-[description]`

---

## 7. Limitations

1. **No real evaluation yet** — All metrics tables are placeholders pending annotation and real pipeline runs.
2. **Single annotator** — Ground truth quality is limited by one reviewer's judgment.
3. **English-only** — Linguistic patterns cover English sponsor language only.
4. **Mock mode produces no real results** — Development testing does not validate detection quality.
5. **IoU threshold choice** — 0.50 is standard but strict; a 90s segment predicted as 60-150s still passes (IoU=0.5), but a 30s segment predicted as 20-60s does not (IoU=10/50=0.2).
6. **Ad type classification accuracy** — Type classification is separate from segment detection. It is possible to detect a segment correctly but classify it as `other` instead of `midroll_sponsor_read`. Evaluation currently conflates these.

---

## 8. Reproducibility Notes

All evaluation results can be reproduced by:

1. Setting the same environment variables (especially providers)
2. Using the same `test_data.json` (exact video URLs/paths)
3. Running `python scripts/run_test_set.py --output results/`
4. Running `python scripts/evaluate.py`

Results will vary between runs if:
- yt-dlp is updated (video quality/format may change)
- OpenAI model versions change (GPT-4o-mini responses are not deterministic)
- Scene detection threshold is changed

For reproducibility, pin `yt-dlp` version and document the OpenAI model version used at evaluation time in the run report.
