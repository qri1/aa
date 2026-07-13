import json
import os
import random
import zipfile
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import CPOConfig, CPOTrainer

SEED = 42
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ARCHIVE_NAME = "ml-olympiad-red-task-c1005bf0-8695-451a-9616-87c8687dce27.zip"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

assert torch.cuda.is_available(), "CUDA GPU is required; do not run this task on CPU"
vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
assert vram_gb >= 7.5, f"This 8GB runner needs at least 8GB VRAM, found {vram_gb:.2f}GB"

root = Path.cwd()
work_root = root / "local_work"
data_root = work_root / "aa_input"
output_root = work_root / "artifacts"
dpo_adapter_root = output_root / "dpo_style_adapter"
assert dpo_adapter_root.exists(), f"Missing {dpo_adapter_root}. Complete tasks 1–4 once before running task 5."

required = ["good_bad.jsonl", "public_test_quality.jsonl"]
if not all((data_root / name).exists() for name in required):
    archives = sorted(root.rglob(ARCHIVE_NAME))
    if len(archives) != 1:
        raise FileNotFoundError(f"Put {ARCHIVE_NAME} in the repository root")
    members = {
        "ml-olympiad-red-task/data/good_bad.jsonl": "good_bad.jsonl",
        "ml-olympiad-red-task/data/public_test_quality.jsonl": "public_test_quality.jsonl",
    }
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as source:
        for member, destination in members.items():
            (data_root / destination).write_bytes(source.read(member))


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


quality_train_rows = read_jsonl(data_root / "good_bad.jsonl")
quality_test_rows = read_jsonl(data_root / "public_test_quality.jsonl")


def quality_prompt(row):
    return row.get("instruction", row.get("prompt"))


simpo_dataset = Dataset.from_list(
    [
        {
            "prompt": [{"role": "user", "content": quality_prompt(row)}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
        for row in quality_train_rows
    ]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization,
    torch_dtype=torch.float16,
    device_map="auto",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)
model = PeftModel.from_pretrained(base_model, dpo_adapter_root, is_trainable=True)
model.config.use_cache = False

args = CPOConfig(
    output_dir=str(work_root / "aa_simpo"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    report_to="none",
    optim="paged_adamw_8bit",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    max_length=224,
    max_prompt_length=96,
    max_completion_length=128,
    beta=2.0,
    loss_type="simpo",
    cpo_alpha=0.0,
    simpo_gamma=1.0,
    seed=SEED,
    data_seed=SEED,
)
trainer = CPOTrainer(
    model=model,
    args=args,
    train_dataset=simpo_dataset,
    processing_class=tokenizer,
)
result = trainer.train()
simpo_adapter_root = output_root / "simpo_adapter"
trainer.model.save_pretrained(simpo_adapter_root)
tokenizer.save_pretrained(simpo_adapter_root)


def mean_answer_logprob(active_model, prompt, answer):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(active_model.device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(active_model.device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.inference_mode():
        logits = active_model(input_ids=input_ids).logits[:, :-1]
    target_ids = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs[:, prompt_ids.shape[1] - 1 :].mean().item())


model = trainer.model
model.eval()
comparisons = []
for row in quality_test_rows:
    prompt = quality_prompt(row)
    comparisons.append(
        {
            "chosen_mean_logprob": mean_answer_logprob(model, prompt, row["chosen"]),
            "rejected_mean_logprob": mean_answer_logprob(model, prompt, row["rejected"]),
        }
    )
accuracy = sum(item["chosen_mean_logprob"] > item["rejected_mean_logprob"] for item in comparisons) / len(comparisons)
answer = "А" if accuracy < 0.6 else "Б" if accuracy < 0.75 else "В" if accuracy < 0.9 else "Г"
(output_root / "simpo_quality_logprobs.json").write_text(
    json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
)
print({"seed": SEED, "gpu": torch.cuda.get_device_name(0), "vram_gb": round(vram_gb, 2), "simpo_train_loss": round(result.training_loss, 6)})
print(f"implicit_preference_accuracy_simpo={accuracy:.6f}")
print(f"answer_simpo={answer}")
