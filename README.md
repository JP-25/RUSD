# RUSD: Reliability–Uncertainty Self-Distillation

Training code for **RUSD**, a self-distillation method that trains an LLM from its own online attempts, using only questions (no reference answers).

One model plays three roles:

- **Student:** answers the question; this answer is the *attempt*.
- **Teacher:** the same model, shown the question together with the complete attempt; it gives token-level targets along the attempt.
- **Judge:** the frozen initial model; it rates the attempt from 1 to 5.

Each attempt is weighted by

$$w = \alpha \, q + (1 - \alpha) \, u, \qquad q = \frac{\text{score} - 1}{4} \times \text{judge confidence},$$

where $q$ is the judged reliability and $u$ is the student's normalized token entropy (its uncertainty). The student is trained with LoRA adapters on the weighted forward-KL distillation loss.

## Repository structure

```
.
├── run_rusd.sh            # launch script (main configuration)
├── rusd_train.py          # entry point: arguments, data selection, W&B, training
├── rusd_trainer.py        # RUSDTrainer: rollouts (vLLM), teacher targets, judge, weights, loss, analysis logs
├── accelerate_rusd.yaml   # Accelerate + DeepSpeed ZeRO-3 configuration
├── config.json            # credentials (fill in, optional)
├── requirements.txt
├── data/
│   ├── models/            # put the base model here
│   └── hf_cache/          # Hugging Face cache for the training data
└── tests/                 # CPU unit tests
```

## Requirements

- 4 GPUs of 80 GB each (NVIDIA A100 80GB)
- CUDA 12.6 or above
- Python 3.10 or above

## Installation

```bash
git clone <this repository>
cd <repository folder>

conda create -n rusd python=3.10 -y
conda activate rusd
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

## Setup

### 1. Credentials (optional)

Fill in `config.json`, or export the same names as environment variables (environment variables take precedence):

```json
{
  "HF_TOKEN": "",
  "WANDB_API_KEY": ""
}
```

- `HF_TOKEN` can stay empty: the models and the dataset are public.
- Without `WANDB_API_KEY`, training logs to Weights & Biases in offline mode under `outputs/`. Set `WANDB_MODE=disabled` to turn W&B off.

### 2. Base model

Download the base model into `data/models/`:

```bash
hf download Qwen/Qwen3-4B --local-dir data/models/Qwen3-4B
# optional, for the 8B run
hf download Qwen/Qwen3-8B --local-dir data/models/Qwen3-8B
```

### 3. Training data

Nothing to do: [NaturalReasoning](https://huggingface.co/datasets/facebook/natural_reasoning) is downloaded automatically on the first run into `data/hf_cache/`. The script keeps rows with a non-empty reference answer, shuffles them with seed 42 and uses the first 30,000 questions. Reference answers are used only for this selection; the student, teacher and judge see only the question and the generated text.

## Training

```bash
bash run_rusd.sh                                              # Qwen3-4B (default)
MODEL_NAME_OR_PATH=./data/models/Qwen3-8B bash run_rusd.sh    # Qwen3-8B
```

Change settings through environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME_OR_PATH` | `./data/models/Qwen3-4B` | Local model folder or Hugging Face model ID |
| `OUTPUT_DIR` | `./outputs` | Checkpoints and logs are written to `OUTPUT_DIR/<run name>/` |
| `NUM_PROCESSES` | `4` | Number of GPUs |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Keep `NUM_PROCESSES × GRADIENT_ACCUMULATION_STEPS = 32` (the effective batch size) |
| `ADDITIVE_QUALITY_MIX` | `0.5` | α: `1` = judged reliability only, `0` = student uncertainty only |
| `TEACHER_MODE` | `dynamic` | `dynamic`: teacher is the current model; `fixed`: teacher is the initial model |
| `MAX_COMPLETION_LENGTH` | `8192` | Maximum response length; truncated responses get zero weight |
| `WANDB_PROJECT` / `WANDB_ENTITY` | `rusd` / your default | W&B destination |
| `AUTO_RESUME` / `RESUME_FROM_CHECKPOINT` | `true` / empty | Resume from the latest (or a given) checkpoint |

Examples:

```bash
ADDITIVE_QUALITY_MIX=0.25 bash run_rusd.sh               # another alpha
NUM_PROCESSES=8 GRADIENT_ACCUMULATION_STEPS=4 bash run_rusd.sh   # 8 GPUs, same batch size
WANDB_MODE=disabled bash run_rusd.sh                      # no W&B logging
```

## Outputs

Everything is written to `OUTPUT_DIR/<run name>/`:

- `checkpoint-<step>/`: LoRA adapter and optimizer state, saved every 50 steps.
- `analysis/samples_rank*_*.jsonl`: one JSON line per training response (step, judge score and its probabilities, judge confidence, student uncertainty, truncation flag, response text).
- `analysis/run_manifest_*.json`: configuration, package versions, and a SHA-256 hash of the selected training data.

## Reproducing the paper

- **Main runs (Qwen3-4B and Qwen3-8B):** run the default script and use **`checkpoint-150`**. The learning rate decays linearly over the full 3-epoch budget, so keep the budget unchanged and stop once `checkpoint-150` is saved. Shortening the budget would change the learning-rate schedule. On 4 A100 80GB GPUs, Qwen3-4B takes about 7.5 hours per 50 steps, so roughly 22 hours to reach `checkpoint-150`.
- **Coefficient study (α):** `ADDITIVE_QUALITY_MIX` ∈ {0, 0.25, 0.5, 0.75, 1} with `TEACHER_MODE=fixed`.

## Acknowledgements

`rusd_trainer.py` builds on the [Hugging Face TRL](https://github.com/huggingface/trl) trainers (Apache-2.0).
