#!/usr/bin/env bash
# Train RUSD (Reliability-Uncertainty Self-Distillation) with LoRA adapters on one node.
# The defaults reproduce the main Qwen3-4B configuration (4 GPUs with 80 GB each).
# Any setting below can be overridden from the environment, for example:
#   MODEL_NAME_OR_PATH=./data/models/Qwen3-8B bash run_rusd.sh
#   ADDITIVE_QUALITY_MIX=0.25 bash run_rusd.sh
set -euo pipefail

cd "$(dirname "$0")"

# ---------------- Paths ----------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./data/models/Qwen3-4B}"   # download the model here first (see README)
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
export HF_HOME="${HF_HOME:-$(pwd)/data/hf_cache}"                    # Hugging Face cache (dataset download)
export WANDB_DIR="${WANDB_DIR:-$(pwd)/outputs}"

# ---------------- Hardware ----------------
NUM_PROCESSES="${NUM_PROCESSES:-4}"                                   # number of GPUs
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"       # 1 question/GPU x 8 x 4 GPUs = 32 per update
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# ---------------- Data ----------------
DATA_SOURCE="${DATA_SOURCE:-natural_reasoning}"
DATASET_NAME="${DATASET_NAME:-facebook/natural_reasoning}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_SAMPLE_SIZE="${DATASET_SAMPLE_SIZE:-30000}"
DATASET_SEED="${DATASET_SEED:-42}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8192}"

# ---------------- Weights & Biases ----------------
WANDB_ENTITY="${WANDB_ENTITY:-}"                                     # empty = your default W&B entity
WANDB_PROJECT="${WANDB_PROJECT:-rusd}"

# ---------------- Response weighting ----------------
# w = a * q + (1 - a) * u, with judged reliability q = ((score - 1) / 4) * judge confidence
# and student uncertainty u. a = ADDITIVE_QUALITY_MIX (0.5 in the main configuration).
JUDGE_QUALITY_WEIGHTING="${JUDGE_QUALITY_WEIGHTING:-true}"
JUDGE_SCORE_WEIGHT_POWER="${JUDGE_SCORE_WEIGHT_POWER:-1.0}"
JUDGE_CONFIDENCE_WEIGHT_POWER="${JUDGE_CONFIDENCE_WEIGHT_POWER:-1.0}"
STUDENT_UNCERTAINTY_WEIGHTING="${STUDENT_UNCERTAINTY_WEIGHTING:-true}"
STUDENT_UNCERTAINTY_MIX="${STUDENT_UNCERTAINTY_MIX:-0.5}"
STUDENT_UNCERTAINTY_WEIGHT_POWER="${STUDENT_UNCERTAINTY_WEIGHT_POWER:-1.0}"
# additive = the weighting above; legacy = the earlier multiplicative design.
RUSD_WEIGHTING_MODE="${RUSD_WEIGHTING_MODE:-additive}"
ADDITIVE_QUALITY_MIX="${ADDITIVE_QUALITY_MIX:-0.5}"
ADDITIVE_QUALITY_THRESHOLD="${ADDITIVE_QUALITY_THRESHOLD:-0.0}"
# dynamic = the teacher is the current model (main configuration); fixed = the initial model.
TEACHER_MODE="${TEACHER_MODE:-dynamic}"
ADDITIVE_ARGS=()
if [[ "${RUSD_WEIGHTING_MODE}" != "additive" && "${RUSD_WEIGHTING_MODE}" != "legacy" ]]; then
    echo "RUSD_WEIGHTING_MODE must be additive or legacy" >&2
    exit 1
fi
case "${TEACHER_MODE}" in
    dynamic) TEACHER_ARGS=(--fixed_teacher false --use_ema_teacher false) ;;
    fixed) TEACHER_ARGS=(--fixed_teacher true --use_ema_teacher false) ;;
    *) echo "TEACHER_MODE must be dynamic or fixed" >&2; exit 1 ;;
esac
STUDENT_UNCERTAINTY_RUN_TAG=""
if [[ "${STUDENT_UNCERTAINTY_WEIGHTING}" == "true" ]]; then
    STUDENT_UNCERTAINTY_RUN_TAG="_studentunc_l${STUDENT_UNCERTAINTY_MIX}_gu${STUDENT_UNCERTAINTY_WEIGHT_POWER}"
    if [[ "${RUSD_WEIGHTING_MODE}" == "additive" ]]; then
        ADDITIVE_ARGS=(--additive_quality_mix "${ADDITIVE_QUALITY_MIX}" --additive_quality_threshold "${ADDITIVE_QUALITY_THRESHOLD}")
        STUDENT_UNCERTAINTY_RUN_TAG="_additive_a${ADDITIVE_QUALITY_MIX}_qmin${ADDITIVE_QUALITY_THRESHOLD}"
        if [[ "${JUDGE_QUALITY_WEIGHTING}" != "true" ]]; then
            echo "Additive weighting requires JUDGE_QUALITY_WEIGHTING=true" >&2
            exit 1
        fi
    fi
fi
MODEL_TAG="$(basename "${MODEL_NAME_OR_PATH:-Qwen3-4B}" | tr '[:upper:]' '[:lower:]')"
RUN_CONFIG="${RUN_CONFIG:-${MODEL_TAG}_rusd_gs${JUDGE_SCORE_WEIGHT_POWER}_gc${JUDGE_CONFIDENCE_WEIGHT_POWER}${STUDENT_UNCERTAINTY_RUN_TAG}_${TEACHER_MODE}_teacher_tok${MAX_COMPLETION_LENGTH:-8192}}"
RUN_OUTPUT_DIR="${OUTPUT_DIR%/}/${RUN_CONFIG}"
echo "Weighting mode: ${RUSD_WEIGHTING_MODE}; additive args: ${ADDITIVE_ARGS[*]:-} (empty = legacy weighting)"

mkdir -p "${RUN_OUTPUT_DIR}" "${WANDB_DIR}"

if [[ ( "${MODEL_NAME_OR_PATH}" == ./* || "${MODEL_NAME_OR_PATH}" == /* ) && ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
    echo "Model not found at ${MODEL_NAME_OR_PATH}. Download it first, e.g." >&2
    echo "  hf download Qwen/Qwen3-4B --local-dir ./data/models/Qwen3-4B" >&2
    exit 1
fi

# W&B: use online logging when an API key is available (environment or config.json), offline otherwise.
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
    CONFIG_WANDB_KEY="$(python -c 'import json; print(json.load(open("config.json")).get("WANDB_API_KEY", ""))' 2>/dev/null || true)"
    if [[ -z "${CONFIG_WANDB_KEY}" ]]; then
        export WANDB_MODE=offline
        echo "No W&B API key found: logging offline to ${WANDB_DIR}"
    fi
fi
WANDB_RUN_ID_FILE="${RUN_OUTPUT_DIR}/wandb_run_id"
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
elif [[ -s "${WANDB_RUN_ID_FILE}" ]]; then
    WANDB_RUN_ID="$(<"${WANDB_RUN_ID_FILE}")"
else
    WANDB_RUN_ID="$(printf '%s' "${RUN_OUTPUT_DIR}-$(date +%s)" | sha256sum | cut -c1-16)"
    printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}"
fi
export WANDB_RUN_ID
export WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

# Resume from the latest checkpoint of this run, if one exists.
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [[ "${AUTO_RESUME:-true}" == "true" ]]; then
    while IFS= read -r checkpoint; do
        if [[ -f "${checkpoint}/trainer_state.json" ]]; then
            RESUME_FROM_CHECKPOINT="${checkpoint}"
            resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
            break
        fi
    done < <(find "${RUN_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -Vr)
fi

echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR}"
echo "TEACHER_MODE=${TEACHER_MODE}  ADDITIVE_QUALITY_MIX=${ADDITIVE_QUALITY_MIX}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-none}"

accelerate launch \
    --config_file accelerate_rusd.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    rusd_train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_source "${DATA_SOURCE}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_sample_size "${DATASET_SAMPLE_SIZE}" \
    --dataset_seed "${DATASET_SEED}" \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 3 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_length 32768 \
    --save_strategy steps \
    --save_steps 50 \
    --save_safetensors true \
    --save_only_model false \
    --logging_steps 2 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --beta 0 \
    --jsd_token_clip 0.05 \
    --use_peft \
    "${TEACHER_ARGS[@]}" \
    --student_thinking true \
    --teacher_thinking true \
    --judge_thinking false \
    --judge_min_score 4 \
    --judge_entropy_threshold 0.3 \
    --judge_temperature 1.0 \
    --judge_max_prompt_length 12288 \
    --judge_quality_weighting "${JUDGE_QUALITY_WEIGHTING}" \
    --judge_score_weight_power "${JUDGE_SCORE_WEIGHT_POWER}" \
    --judge_confidence_weight_power "${JUDGE_CONFIDENCE_WEIGHT_POWER}" \
    --student_uncertainty_weighting "${STUDENT_UNCERTAINTY_WEIGHTING}" \
    --student_uncertainty_mix "${STUDENT_UNCERTAINTY_MIX}" \
    --student_uncertainty_weight_power "${STUDENT_UNCERTAINTY_WEIGHT_POWER}" \
    ${ADDITIVE_ARGS[@]+"${ADDITIVE_ARGS[@]}"} \
    --skip_truncated_rollouts true \
    --rusd_objective positive_only \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_tensor_parallel_size 1 \
    --temperature 1.2 \
    --top_p 0.95 \
    --top_k 20 \
    "${WANDB_ARGS[@]}" \
    ${resume_args[@]+"${resume_args[@]}"}
