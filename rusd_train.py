import json
import glob
import hashlib
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
import wandb
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.gold import GOLDConfig

from rusd_trainer import RUSDTrainer, save_manifest


RUBRICHUB_SCIENCE_SOURCES = {"RaR_science.jsonl", "MegeScience.jsonl"}
RURL_SCIENCE_TRAIN_DIR = os.environ.get("RURL_SCIENCE_TRAIN_DIR", "data/rurl_science/train")


def setup_auth(config_path="config.json"):
    """Load local credentials without overwriting credentials exported by Slurm."""
    needs_hf_token = not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    needs_wandb_key = not os.environ.get("WANDB_API_KEY")
    if not (needs_hf_token or needs_wandb_key):
        return
    if not os.path.isfile(config_path):
        missing = []
        if needs_hf_token:
            missing.append("HF_TOKEN")
        if needs_wandb_key:
            missing.append("WANDB_API_KEY")
        raise RuntimeError(f"Missing {', '.join(missing)} and {config_path!r} was not found.")

    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if needs_hf_token and config.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = config["HF_TOKEN"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = config["HF_TOKEN"]
    if needs_wandb_key and config.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]


def _question_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("content"):
                parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return str(value.get("content", value)).strip()
    return "" if value is None else str(value).strip()


def _normalize_question(example):
    for key in ("question", "problem", "prompt", "instruction", "input", "query"):
        if key in example and example[key] is not None:
            return {"question": _question_text(example[key])}
    return {"question": ""}


def _first_text(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return _question_text(example[key])
    return ""


def _first_value(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _selection_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_data_source(value):
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "natural": "natural_reasoning",
        "naturalreasoning": "natural_reasoning",
        "rar": "rar_science",
        "rarscience": "rar_science",
        "rubric_hub": "rubrichub",
        "rurl": "rurl_science",
        "rurlscience": "rurl_science",
        "openthoughts": "openthoughts_math",
        "custom_dataset": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {
        "natural_reasoning",
        "rar_science",
        "rubrichub",
        "rurl_science",
        "openthoughts_math",
        "custom",
    }
    if normalized not in valid:
        raise ValueError(f"Unsupported data_source={value!r}; expected one of {sorted(valid)}.")
    return normalized


def _rubrichub_parquets():
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    patterns = [
        os.path.join(root, "datasets--sojuL--RubricHub_v1", "snapshots", "*", "sft_RuFT", "*.parquet")
        for root in dict.fromkeys(root for root in roots if root)
    ]
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if files:
        return files
    snapshot = snapshot_download(
        "sojuL/RubricHub_v1",
        repo_type="dataset",
        allow_patterns=["sft_RuFT/*.parquet"],
    )
    return sorted(glob.glob(os.path.join(snapshot, "sft_RuFT", "*.parquet")))


def _load_rubrichub_dataset(sample_size, seed):
    files = _rubrichub_parquets()
    random.Random(seed).shuffle(files)
    rows = []
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        required = {"source", "query", "answer", "rubrics"}
        if not required.issubset(parquet.schema_arrow.names):
            continue
        remaining_files = len(files) - file_index
        file_limit = None
        if sample_size > 0:
            remaining_rows = sample_size - len(rows)
            if remaining_rows <= 0:
                break
            file_limit = max(1, (remaining_rows + remaining_files - 1) // remaining_files)
        file_rows = 0
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=1024):
            for example in batch.to_pylist():
                if example.get("source") not in RUBRICHUB_SCIENCE_SOURCES:
                    continue
                question = _question_text(example.get("query"))
                reference = _question_text(example.get("answer"))
                if not question or not reference or not example.get("rubrics"):
                    continue
                rows.append({"question": question, "reference_answer": reference})
                file_rows += 1
                if sample_size > 0 and (len(rows) >= sample_size or file_rows >= file_limit):
                    break
            if sample_size > 0 and (len(rows) >= sample_size or file_rows >= file_limit):
                break
    if not rows:
        raise RuntimeError("No usable RubricHub Science rows were found.")
    return Dataset.from_list(rows)


def _load_training_dataset(script_args):
    source = script_args.data_source
    if source == "natural_reasoning":
        return load_dataset("facebook/natural_reasoning", split="train")
    if source == "rar_science":
        return load_dataset("anisha2102/RaR-Science", split="train")
    if source == "rubrichub":
        return _load_rubrichub_dataset(script_args.dataset_sample_size, script_args.dataset_seed)
    if source == "rurl_science":
        if not os.path.isdir(RURL_SCIENCE_TRAIN_DIR):
            raise RuntimeError(
                f"RuRL Science data not found at {RURL_SCIENCE_TRAIN_DIR}; run preprocess_rurl_science.py first."
            )
        return load_from_disk(RURL_SCIENCE_TRAIN_DIR)
    if source == "openthoughts_math":
        return load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")

    load_args = [script_args.dataset_name]
    if script_args.dataset_config:
        load_args.append(script_args.dataset_config)
    return load_dataset(*load_args, split=script_args.dataset_split)


def _has_reference_answer(example):
    reference = _first_value(
        example,
        ("reference_answer", "reference", "answer", "final_answer", "solution"),
    )
    return bool(_selection_text(reference).strip())


def _selection_fingerprint(dataset):
    digest = hashlib.sha256()
    for example in dataset:
        question = _first_value(
            example, ("question", "problem", "prompt", "instruction", "input", "query")
        )
        reference = _first_value(
            example,
            ("reference_answer", "reference", "answer", "final_answer", "solution"),
        )
        digest.update(_selection_text(question).strip().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_selection_text(reference).strip().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class RUSDScriptArguments(ScriptArguments):
    data_source: str = field(
        default="natural_reasoning",
        metadata={
            "help": "Dataset source: natural_reasoning, rar_science, rubrichub, "
            "rurl_science, openthoughts_math, or custom."
        },
    )
    dataset_name: str = field(default="facebook/natural_reasoning")
    dataset_config: str | None = field(default=None)
    dataset_split: str = field(default="train")
    dataset_sample_size: int = field(
        default=30000,
        metadata={"help": "Number of shuffled training questions; 0 uses the full split."},
    )
    dataset_seed: int = field(default=42)
    run_config: str | None = field(default=None)
    presence_penalty: float = field(default=0.0)
    fixed_teacher: bool = field(default=True)
    use_ema_teacher: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    student_thinking: bool = field(default=True)
    teacher_thinking: bool = field(default=True)
    judge_thinking: bool = field(
        default=False,
        metadata={"help": "Must remain false for the current score-only next-token judge."},
    )
    judge_min_score: int = field(default=4)
    judge_entropy_threshold: float = field(
        default=0.5,
        metadata={"help": "Maximum normalized entropy in [0, 1] for a trusted judgment."},
    )
    judge_temperature: float = field(default=1.0)
    judge_max_prompt_length: int = field(default=8192)
    judge_quality_weighting: bool = field(
        default=False,
        metadata={
            "help": "Replace the hard judge gate with q=((score-1)/4)^gamma_s*(1-H/log5)^gamma_c."
        },
    )
    judge_score_weight_power: float = field(default=1.0)
    judge_confidence_weight_power: float = field(default=1.0)
    student_uncertainty_weighting: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable student uncertainty weighting. With additive_quality_mix=a, use a*q+(1-a)*u. Otherwise multiply q by "
                "(1-lambda)+lambda*u_student^gamma, where u_student is the detached "
                "response-token entropy normalized by log(vocabulary_size)."
            )
        },
    )
    student_uncertainty_mix: float = field(default=0.5)
    student_uncertainty_weight_power: float = field(default=1.0)
    additive_quality_mix: Optional[float] = field(
        default=None,
        metadata={"help": "Use w=a*q+(1-a)*u with this quality coefficient a in [0,1]. Requires both weighting flags and unit powers; omitting retains legacy weighting."},
    )
    additive_quality_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum original judge quality x for additive weighting, in [0,1]; 0 preserves ungated behavior."},
    )
    skip_truncated_rollouts: bool = field(
        default=True,
        metadata={
            "help": "Exclude responses that reach max_completion_length without an earlier stop from distillation."
        },
    )
    rusd_objective: str = field(
        default="positive_only",
        metadata={"help": "RUSD loss mode; only positive_only is implemented."},
    )
    jsd_token_clip: float = field(default=0.0)
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Positive-only RUSD: teacher top-k renormalized loss; 0 uses full vocabulary. Does not affect generation or judging."},
    )


def _resolve_dtype(model_args):
    value = getattr(model_args, "torch_dtype", None) or getattr(model_args, "dtype", None)
    if not isinstance(value, str):
        return value or torch.bfloat16
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(value.lower(), torch.bfloat16)


if __name__ == "__main__":
    setup_auth()
    parser = TrlParser((RUSDScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if script_args.top_k_loss < 0:
        raise ValueError("top_k_loss must be nonnegative; 0 uses full vocabulary.")
    script_args.data_source = _normalize_data_source(script_args.data_source)
    if not 0.0 <= script_args.additive_quality_threshold <= 1.0:
        raise ValueError("additive_quality_threshold must be in [0, 1].")
    if script_args.additive_quality_threshold > 0 and script_args.additive_quality_mix is None:
        raise ValueError("A nonzero additive_quality_threshold requires additive_quality_mix.")

    if not model_args.use_peft:
        raise ValueError("RUSD currently requires --use_peft so the judge can remain the fixed initial model.")
    if script_args.fixed_teacher and script_args.use_ema_teacher:
        raise ValueError("fixed_teacher and use_ema_teacher are mutually exclusive.")
    if script_args.rusd_objective != "positive_only":
        raise ValueError(
            f"Unsupported rusd_objective={script_args.rusd_objective!r}; only 'positive_only' is implemented."
        )

    run_config = script_args.run_config or (
        f"rusd_{script_args.rusd_objective}_{Path(model_args.model_name_or_path).name}_"
        f"b{training_args.beta}_"
        f"score{script_args.judge_min_score}_entropy{script_args.judge_entropy_threshold}"
        + (f"_additive_a{script_args.additive_quality_mix}" if script_args.additive_quality_mix is not None else "")
        + (f"_qmin{script_args.additive_quality_threshold}_eps1e-6" if script_args.additive_quality_threshold > 0 else "")
    )
    training_args.output_dir = training_args.output_dir or "outputs/rusd"
    if not training_args.output_dir.endswith(run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / run_config)

    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb_kwargs = dict(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=run_config,
            config={
                "method": "reliability_uncertainty_self_distillation",
                "model_name": model_args.model_name_or_path,
                "data_source": script_args.data_source,
                "dataset_name": script_args.dataset_name,
                "dataset_config": script_args.dataset_config,
                "dataset_split": script_args.dataset_split,
                "dataset_sample_size": script_args.dataset_sample_size,
                "dataset_seed": script_args.dataset_seed,
                "effective_batch_size": effective_batch_size,
                "beta": training_args.beta,
                "top_k_loss": script_args.top_k_loss,
                "loss_vocabulary": (f"renormalized_teacher_top_{script_args.top_k_loss}"
                                    if script_args.top_k_loss > 0 else "full"),
                "fixed_teacher": script_args.fixed_teacher,
                "use_ema_teacher": script_args.use_ema_teacher,
                "judge_strategy": "fixed_initial_model",
                "judge_thinking": script_args.judge_thinking,
                "judge_min_score": script_args.judge_min_score,
                "judge_entropy_threshold": script_args.judge_entropy_threshold,
                "judge_temperature": script_args.judge_temperature,
                "judge_quality_weighting": script_args.judge_quality_weighting,
                "judge_score_weight_power": script_args.judge_score_weight_power,
                "judge_confidence_weight_power": script_args.judge_confidence_weight_power,
                "student_uncertainty_weighting": script_args.student_uncertainty_weighting,
                "student_uncertainty_mix": script_args.student_uncertainty_mix,
                "student_uncertainty_weight_power": script_args.student_uncertainty_weight_power,
                "additive_quality_mix": script_args.additive_quality_mix,
                "additive_quality_threshold": script_args.additive_quality_threshold,
                "additive_rejected_weight": 1e-6 if script_args.additive_quality_mix is not None else None,
                "skip_truncated_rollouts": script_args.skip_truncated_rollouts,
                "rusd_objective": script_args.rusd_objective,
            },
        )
        if os.environ.get("WANDB_RUN_ID"):
            wandb_kwargs["id"] = os.environ["WANDB_RUN_ID"]
            wandb_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
        wandb.init(**wandb_kwargs)

    model_dtype = _resolve_dtype(model_args)
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation or "flash_attention_2",
        "torch_dtype": model_dtype,
        "use_cache": not training_args.gradient_checkpointing,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs
    training_args.dataset_text_field = "question"
    training_args.presence_penalty = script_args.presence_penalty

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _load_training_dataset(script_args)
    train_dataset = train_dataset.filter(
        _has_reference_answer,
        desc="Filtering empty reference answers",
    )
    train_dataset = train_dataset.shuffle(seed=script_args.dataset_seed)
    if 0 < script_args.dataset_sample_size < len(train_dataset):
        train_dataset = train_dataset.select(range(script_args.dataset_sample_size))
    selection_sha256 = _selection_fingerprint(train_dataset)

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        _normalize_question,
        remove_columns=original_columns,
        desc="Keeping reference-free questions only",
    )
    empty_questions = sum(not question.strip() for question in train_dataset["question"])
    if empty_questions:
        raise RuntimeError(
            f"The deterministic selection contains {empty_questions} empty questions. "
            "Refusing to remove or replace rows because that would diverge from RCSD's selected dataset."
        )
    print("\n" + "=" * 80)
    print(f"RUSD run: {run_config}")
    print(f"Output: {training_args.output_dir}")
    print(f"Reference-free training questions: {len(train_dataset)}")
    print(f"Selected dataset SHA256: {selection_sha256}")
    loss_vocabulary = (f"renormalized teacher top-{script_args.top_k_loss}"
                       if script_args.top_k_loss > 0 else "full vocabulary")
    print(f"Loss: beta={training_args.beta}, {loss_vocabulary}")
    print("=" * 80 + "\n")
    if os.environ.get("LOCAL_RANK", "0") == "0" and wandb.run is not None:
        wandb.config.update(
            {
                "selected_dataset_size": len(train_dataset),
                "selected_dataset_sha256": selection_sha256,
            },
            allow_val_change=True,
        )

    trainer_kwargs = dict(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        judge_thinking=script_args.judge_thinking,
        judge_min_score=script_args.judge_min_score,
        judge_entropy_threshold=script_args.judge_entropy_threshold,
        judge_temperature=script_args.judge_temperature,
        judge_max_prompt_length=script_args.judge_max_prompt_length,
        judge_quality_weighting=script_args.judge_quality_weighting,
        judge_score_weight_power=script_args.judge_score_weight_power,
        judge_confidence_weight_power=script_args.judge_confidence_weight_power,
        student_uncertainty_weighting=script_args.student_uncertainty_weighting,
        student_uncertainty_mix=script_args.student_uncertainty_mix,
        student_uncertainty_weight_power=script_args.student_uncertainty_weight_power,
        additive_quality_mix=script_args.additive_quality_mix,
        additive_quality_threshold=script_args.additive_quality_threshold,
        skip_truncated_rollouts=script_args.skip_truncated_rollouts,
        rusd_objective=script_args.rusd_objective,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        top_k_loss=script_args.top_k_loss or None,
    )
    trainer = RUSDTrainer(**trainer_kwargs)
    # Save the run manifest (configuration, package versions, data hash) and tag analysis records with it.
    manifest_paths = [None]
    if trainer.accelerator.is_main_process:
        manifest_paths[0] = save_manifest(
            training_args.output_dir,
            {"script_args": asdict(script_args), "training_args": training_args.to_dict(),
             "model_args": asdict(model_args)}, selection_sha256)
        print(f"RUSD analysis manifest: {manifest_paths[0]}")
    broadcast_object_list(manifest_paths)
    trainer._analysis_manifest_name = Path(manifest_paths[0]).name
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
