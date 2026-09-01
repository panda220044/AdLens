# AdLens — Evaluation Framework & Methodology Specification

## 1. Evaluation Objective

The objective of the AdLens evaluation framework is to rigorously measure the accuracy, boundary precision, and computational efficiency of the multimodal advertisement detection pipeline across diverse video formats (long-form YouTube VODs, YouTube Shorts, and Instagram Reels).

The evaluation suite measures:
- **Segment Detection Accuracy**: Precision, Recall, and F1 score at the advertisement segment level.
- **Temporal Alignment**: Intersection over Union (IoU) and Root Mean Squared Error (RMSE) for start and end timestamps.
- **Taxonomy Classification Accuracy**: Accuracy of assigned ad categories (`preroll`, `midroll_sponsor_read`, `self_promo`, etc.).
- **Resource & Cost Efficiency**: Execution wall-clock latency (`wall_clock_s`) and estimated API cost (`estimated_cost_usd`).

---

## 2. Test Dataset Specification

The evaluation dataset comprises the 5 benchmark videos defined in `test_data.json`:

| Video ID | Source / Platform | Format | Category / Description | Ingestion Source |
|---|---|---|---|---|
| `yt_ujFWRFYLGjY` | YouTube | Long-form VOD | Tech review video with midroll sponsor read | YouTube URL |
| `yt_shorts_Ve0zdhTQA4U` | YouTube Shorts | Short-form | Short-form vertical promotional video | YouTube Shorts URL |
| `yt_s0LLVQeMmtU` | YouTube | Long-form VOD | Educational content with creator self-promo & affiliate read | YouTube URL |
| `ig_reel_DJl6-v8oufg` | Instagram Reels | Short-form | Commercial Instagram Reel promotion | Local file path (`.mp4`) |
| `ig_reel_Db78RoIuOyl` | Instagram Reels | Short-form | Product showcase Instagram Reel | Local file path (`.mp4`) |

*Note: As detailed in the README and DESIGN.md, Instagram Reels require manual local downloading to comply with anti-bot policies.*

---

## 3. Ground Truth Annotation Methodology

Ground truth annotations define the exact start and end timestamps ($\text{start\_s}, \text{end\_s}$), ad taxonomy type, and brand for every commercial segment in the test dataset.

### Annotation Rules
1. **Start Timestamp ($\text{start\_s}$)**: The exact frame where the creator initiates a sponsor disclosure, holds up a sponsored product, or where a promotional graphic first appears.
2. **End Timestamp ($\text{end\_s}$)**: The exact frame where the creator transitions back to organic video content or where promotional lower-thirds disappear.
3. **Ambiguity Rulings**: Annotators enforce Rulings 1–8 (e.g. self-promotion is annotated as `self_promo`; ambient logo wear is not annotated).

### Annotation Tool (`adlens/evaluation/annotate.py`)
AdLens provides an interactive CLI tool for step-by-step annotation:
```powershell
python -m adlens.evaluation.annotate --output ground_truth.json
```
The tool iterates through each test video, prompts for segment start/end times, taxonomy type, and brand, and saves formatted ground truth structures to `ground_truth.json`.

---

## 4. Evaluation Metrics Definition

### A. Segment Intersection over Union (IoU)
For a predicted segment $P = [t_1^{\text{pred}}, t_2^{\text{pred}}]$ and a ground truth segment $G = [t_1^{\text{gt}}, t_2^{\text{gt}}]$:

$$\text{IoU}(P, G) = \frac{\text{Intersection}(P, G)}{\text{Union}(P, G)} = \frac{\max\left(0, \min(t_2^{\text{pred}}, t_2^{\text{gt}}) - \max(t_1^{\text{pred}}, t_1^{\text{gt}})\right)}{\max(t_2^{\text{pred}}, t_2^{\text{gt}}) - \min(t_1^{\text{pred}}, t_1^{\text{gt}})}$$

### B. Segment Matching Criteria
A predicted segment $P$ matches a ground truth segment $G$ if:
1. $\text{IoU}(P, G) \ge 0.50$ (IoU threshold).
2. Greedy bipartite matching pairs predictions to ground truth items in descending order of IoU. Each ground truth and prediction item can be matched at most once.

### C. Precision, Recall, and F1 Score
- **True Positive (TP)**: Matched prediction with $\text{IoU} \ge 0.50$.
- **False Positive (FP)**: Predicted segment with no matching ground truth segment ($\text{IoU} < 0.50$).
- **False Negative (FN)**: Ground truth segment missed by all predictions.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1 Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### D. Boundary Error (Timestamp Accuracy)
For matched pairs $(P_k, G_k)$:
- **Start Error**: $|t_{1,k}^{\text{pred}} - t_{1,k}^{\text{gt}}|$ (in seconds).
- **End Error**: $|t_{2,k}^{\text{pred}} - t_{2,k}^{\text{gt}}|$ (in seconds).
- **Mean Boundary Error**: Average start and end timestamp error across all true positive matches.

---

## 5. Evaluation Execution Workflow

The evaluation pipeline is executed in 3 standardized steps:

### Step 1: Execute Test Set Batch Analysis
```powershell
python scripts/run_test_set.py --output results/
```
Processes all 5 test videos and generates individual JSON outputs (`results/yt_ujFWRFYLGjY.json`, etc.) and combined predictions (`results/all_results.json`).

### Step 2: Interactive Ground Truth Creation (if not already annotated)
```powershell
python -m adlens.evaluation.annotate --output ground_truth.json
```

### Step 3: Compute Evaluation Metrics & Generate Report
```powershell
python scripts/evaluate.py --predictions results/all_results.json --ground-truth ground_truth.json --output eval_report.txt --json-output eval_results.json
```

---

## 6. Verification & Evaluation Results Status

### Evaluation Infrastructure Status: VERIFIED
The evaluation core code (`adlens/evaluation/__init__.py`), matching algorithms, IoU functions, metric aggregations, and CLI scripts are **100% verified and tested** via unit tests (`tests/test_evaluation.py` passed 15/15 tests).

### Ground Truth Dataset Status: PENDING ANNOTATION
To maintain strict scientific integrity and adhere to non-fabrication guidelines:
- **Actual video ground truth numbers are marked as pending** until manual viewing and interactive annotation (`ground_truth.json`) is completed for the 5 benchmark videos.
- No fake precision, recall, or F1 accuracy scores have been invented.

---

## 7. Performance & Cost Instrumentation

The system tracks resource metrics across all test runs:

| Statistic | Field Name | Description |
|---|---|---|
| Wall-Clock Time | `wall_clock_s` | Total execution duration from ingestion to response. |
| Estimated Cost | `estimated_cost_usd` | Total API cost based on Whisper and GPT-4o-mini rates ($0.00$ in free local mode). |
| Frames Sampled | `frames_sampled` | Number of video frames decoded and analyzed. |
| Model Calls | `model_calls` | Number of external API or local ML inferences performed. |

---

## 8. Error Analysis & Failure Modes

During evaluation, potential mismatches are grouped into four failure categories:

1. **Boundary Misalignment (Low IoU)**:
   * *Symptom*: Prediction overlaps ground truth but $\text{IoU} < 0.50$.
   * *Cause*: Sponsor read intro banter blends into main video topic without a clear visual scene cut.
2. **Missed Visual Placement (False Negative)**:
   * *Symptom*: Product placement or logo display is missed.
   * *Cause*: Product is displayed silently without transcript keywords or on-screen OCR text.
3. **Over-Segmentation (False Positive)**:
   * *Symptom*: Single sponsor read split into two segments.
   * *Cause*: Creator pauses speech for $>3.0\text{s}$ during host read while visuals change. Handled by Rule #8 merging when brand matches.
4. **Platform Ads (Non-Extractable)**:
   * *Symptom*: Ground truth annotator saw a pre-roll on YouTube website, but prediction missed it.
   * *Cause*: Injected platform ads do not exist in downloaded `.mp4` stream.

---

## 9. Reproducibility

All evaluation metrics are fully reproducible:
- Ground truth file format is standard JSON schema (`ground_truth.json`).
- Evaluation code (`scripts/evaluate.py`) is deterministic.
- Unit tests (`python -m pytest tests/test_evaluation.py -v`) verify metric calculation correctness.
