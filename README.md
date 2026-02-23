# CF_OpenGait — Frailty Gait Classification

A framework for classifying frailty levels (Frail, Prefrail, Nonfrail) from gait silhouette sequences, built on [OpenGait](https://github.com/ShiqiYu/OpenGait). Supports **DeepGaitV2** (CNN-based) and **SwinGait** (CNN + Transformer) architectures with configurable layer freezing strategies and k-fold cross-validation.

---

## Prerequisites

- Python 3.8+
- CUDA-capable GPU(s)
- Conda (recommended)

---

## Step 1: Clone the Repo

```bash
git clone https://github.com/lauramcdaniel006/CF_OpenGait.git
cd CF_OpenGait
```

---

## Step 2: Install Dependencies

**Dependencies:** pytorch >= 1.10, torchvision, pyyaml, tensorboard, opencv-python, tqdm, py7zr, kornia, einops

**Install by Anaconda:**

```bash
conda create -n myGait38 python=3.8
conda activate myGait38
conda install tqdm pyyaml tensorboard opencv kornia einops -c conda-forge
conda install pytorch==1.10 torchvision -c pytorch
```

**Or, install by pip:**

```bash
pip install tqdm pyyaml tensorboard opencv-python kornia einops
pip install torch==1.10 torchvision==0.11
```

---

## Step 3: Download and Prepare the Frailty Dataset

Download the frailty silhouette dataset from: **[DATASET_URL]**

The silhouette data must be organized as `.pkl` files in the following directory structure:

```
sils/
├── 300/                            ← subject ID
│   └── Prefrail/                   ← frailty label (Frail / Prefrail / Nonfrail)
│       └── silhouettes/
│           └── silhouettes.pkl     ← silhouette sequence for this subject
├── 310/
│   └── Nonfrail/
│       └── silhouettes/
│           └── silhouettes.pkl
├── 352/
│   └── Frail/
│       └── silhouettes/
│           └── silhouettes.pkl
└── ...
```

Each subject has **one** frailty label directory (`Frail`, `Prefrail`, or `Nonfrail`) containing a `silhouettes/silhouettes.pkl` file with the full silhouette sequence.

**Frailty labels** — already included at `opengait/frailty_label.csv`.
Maps subject IDs → frailty scores (0 = Nonfrail, 1 = Prefrail, 2 = Frail).

---

## Step 4: Download Pretrained Weights

Two pretrained checkpoints are needed (from original OpenGait training on the CCPG dataset):

| Model | File | Hugging Face Link |
|---|---|---|
| DeepGaitV2 | `DeepGaitV2-60000.pt` | [Download](https://huggingface.co/opengait/OpenGait/tree/main/CCPG/DeepGaitV2/DeepGaitV2/checkpoints) |
| SwinGait | `SwinGait3D_B1122_C2-20000.pt` | [Download](https://huggingface.co/opengait/OpenGait/tree/main/CCPG/SwinGait/SwinGait3D_B1122_C2/checkpoints) |

---

## Step 5: Update the Config File

Edit the config YAML (e.g., `configs/deepgaitv2/DeepGaitV2_D3.yaml`):

```yaml
data_cfg:
  dataset_root: /your/path/to/sils                    # ← your silhouette directory
  frailty_label_file: opengait/frailty_label.csv       # ← frailty labels (included)

evaluator_cfg:
  restore_hint: /your/path/to/DeepGaitV2-60000.pt     # ← pretrained weights

trainer_cfg:
  restore_hint: /your/path/to/DeepGaitV2-60000.pt     # ← same pretrained weights
```

### Freezing Strategies

Freezing controls which pretrained layers stay fixed vs. fine-tune for frailty classification. The classification head (FCs + BNNecks) is always trainable.

**DeepGaitV2** — 5 CNN blocks (`layer0`–`layer4`), set in `model_cfg.Backbone.freeze_layers`:

| Config | `freeze_layers` | Description |
|---|---|---|
| `DeepGaitV2_D0` | `false` | All layers trainable |
| `DeepGaitV2_D1` | `[0]` | Layer 0 frozen |
| `DeepGaitV2_D2` | `[0, 1]` | Layers 0–1 frozen |
| `DeepGaitV2_D3` | `[0, 1, 2]` | Layers 0–2 frozen |
| `DeepGaitV2_D4` | `[0, 1, 2, 3]` | Layers 0–3 frozen |
| `DeepGaitV2_D5` | `true` | All CNN layers frozen |
| `DeepGaitV2_D0_with_weights` | `false` + class weights | All trainable + weighted loss |
| `DeepGaitV2_D1_with_weights` | `[0, 1, 2]` + class weights | Layers 0–2 frozen + weighted loss |

**SwinGait** — CNN + Transformer, controlled separately via `freeze_layers` (CNN) and `frozen_stages` (Transformer):

| Config | CNN (`freeze_layers`) | Transformer (`frozen_stages`) | Description |
|---|---|---|---|
| `SwinGait_M1` | `false` | `-1` | Fully unfrozen |
| `SwinGait_M2` | `true` | `-1` | CNN frozen, transformer trainable |
| `SwinGait_M3` | `true` | `0` | CNN frozen + patch embedding frozen |
| `SwinGait_M4` | `true` | `1` | CNN frozen + Stage 1 frozen |
| `SwinGait_M5` | `true` | `2` | CNN frozen + Stages 1–2 frozen |
| `SwinGait_M1_with_weights` | `false` + class weights | `-1` | Fully unfrozen + weighted loss |
| `SwinGait_M2_with_weights` | `true` + class weights | `-1` | CNN frozen + weighted loss |

---

## Step 6: Run K-Fold Cross-Validation

```bash
# DeepGaitV2 example (D3 = layers 0-2 frozen)
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_D3.yaml \
  --k 5 \
  --device 0,1 \
  --nproc 2 \
  --use-existing-partitions

# SwinGait example (M2 = CNN frozen, transformer trainable)
python run_kfold_cross_validation.py \
  --config configs/swingait/SwinGait_M2.yaml \
  --k 5 \
  --device 0,1 \
  --nproc 2 \
  --use-existing-partitions
```

This script:
1. Loads partition files from `kfold_results/partitions/fold_1.json` through `fold_5.json`
2. For each fold, rewrites the config to point at that fold's train/test split
3. Trains the model and evaluates on the test set
4. Saves checkpoints every 500 iterations

### Key Flags

| Flag | Description |
|---|---|
| `--config` | Path to config YAML |
| `--k` | Number of folds (default: 5) |
| `--device` | CUDA devices (default: `0,1`) |
| `--nproc` | Number of processes for distributed training (default: 2) |
| `--use-existing-partitions` | Reuse `kfold_results/partitions/fold_*.json` instead of creating new splits |
| `--skip_training` | Skip training, only run evaluation (assumes checkpoints exist) |
| `--checkpoint_iter` | Evaluate a specific checkpoint iteration (default: auto-selects best) |
| `--best_metric` | Metric for auto-selecting best checkpoint: `accuracy`, `auc_macro`, `f1`, `precision`, `recall` |

---

## Step 7: Aggregate Results Across Folds

After training completes, aggregate metrics across all k folds.

**1. Identify the best checkpoint iteration** for each fold by reviewing training logs in `output/<save_name>/`.

**2. Edit the aggregation script** — update `MODEL_CONFIGS` with the best iteration per fold:

```python
MODEL_CONFIGS = {
    'D3': {   # Example: DeepGaitV2 with layers 0-2 frozen
        1: 500,    # best iteration for fold 1
        2: 4500,   # best iteration for fold 2
        3: 6500,   # best iteration for fold 3
        4: 7000,   # best iteration for fold 4
        5: 8500,   # best iteration for fold 5
    }
}
```

**3. Update `MODEL_DIR_PATTERNS`** to match the `save_name` from your config:

```python
MODEL_DIR_PATTERNS = {
    'D3': 'your_save_name_fold'
}
```

**4. Run:**

```bash
python aggregate_deepgaitv2_metrics.py   # For DeepGaitV2 models
python aggregate_swingait_metrics.py     # For SwinGait models
```

The script parses training logs, extracts metrics at the specified iterations, and outputs mean ± std across folds for accuracy, AUC, F1, precision, recall, and more.

---

## Project Structure

```
CF_OpenGait/
├── configs/
│   ├── deepgaitv2/          ← DeepGaitV2 config files
│   ├── swingait/            ← SwinGait config files
│   └── default.yaml         ← Default config (merged with model configs)
├── kfold_results/
│   └── partitions/          ← Fold partition files (fold_1.json – fold_5.json)
├── opengait/
│   ├── data/                ← Dataset loading
│   ├── evaluation/          ← Evaluation metrics (evaluator.py)
│   ├── modeling/
│   │   ├── models/
│   │   │   ├── deepgaitv2.py   ← DeepGaitV2 model
│   │   │   └── swingait.py     ← SwinGait model
│   │   ├── base_model.py       ← Base model class (training/testing loops)
│   │   └── losses/             ← Loss functions (CE, Triplet, Focal, etc.)
│   ├── utils/               ← Utilities (config loading, logging, etc.)
│   ├── main.py              ← Entry point for training/testing
│   └── frailty_label.csv    ← Frailty labels for all subjects
├── run_kfold_cross_validation.py   ← K-fold CV orchestrator
├── aggregate_deepgaitv2_metrics.py ← Aggregate DeepGaitV2 results
└── aggregate_swingait_metrics.py   ← Aggregate SwinGait results
```
