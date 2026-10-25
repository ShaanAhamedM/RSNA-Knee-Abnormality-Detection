# RSNA Knee Abnormality Detection — Kaggle Submission Package

This directory contains the complete, self-contained offline inference package for the **RSNA Knee Abnormality Detection** Kaggle competition.

---

## Package Contents

1. **`kaggle_inference_notebook.ipynb`**: Complete Jupyter notebook ready to upload directly to Kaggle Notebooks.
2. **`kaggle_inference_notebook.py`**: Standalone Python script version of the complete inference engine.
3. **`bundle_submission_weights.py`**: Automated script that converts all elite PyTorch model weights to FP16 and packs them into `best_ensemble_weights_fp16.zip`.
4. **`best_ensemble_weights_fp16.zip`**: Bundled weights archive containing the trained checkpoints and `grandmaster_params.json`.

---

## 3-Step Submission Instructions to Kaggle

### Step 1: Create the Private Weights Dataset on Kaggle
1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
2. Title the dataset: `rsna-knee-weights`.
3. Drag and drop `best_ensemble_weights_fp16.zip` (located in this folder).
4. Click **Create**. Kaggle will automatically unzip the weights into `/kaggle/input/rsna-knee-weights/`.

### Step 2: Create the Kaggle Inference Notebook
1. Go to the [RSNA Knee Abnormality Detection Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code).
2. Click **New Notebook**.
3. Under **Notebook Options** (right sidebar):
   - **Accelerator:** GPU (NVIDIA T4 x2 or P100).
   - **Internet:** Off (Required by Kaggle submission rules).
   - **Data:** Add Competition Data (`rsna-knee-abnormality-detection`) and Add your newly created dataset (`rsna-knee-weights`).
4. Click **File** -> **Upload Notebook** -> Select `kaggle_inference_notebook.ipynb`.

### Step 3: Run and Submit
1. Click **Save Version** -> Select **Save & Run All (Commit)**.
2. Once the notebook finishes committing (typically <15 minutes on test set):
3. Click the **Output** tab -> Locate `submission.csv` -> Click **Submit to Competition**!
