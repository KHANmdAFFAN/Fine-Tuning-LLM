import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


DEFAULT_CONFIG: Dict[str, Any] = {
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    "project_dir": os.path.abspath(BASE_DIR, "project_outputs")),
    "data_path": os.path.abspath(os.path.join(os.getcwd(), "data", "hr_policy_finetune_1.jsonl")),
    "stage1_adapter": os.path.abspath(os.path.join(os.getcwd(), "project_outputs", "Adapter", "non_instruction_self_hr_policy_adapter")),
    "merged_stage1": os.path.abspath(os.path.join(os.getcwd(), "project_outputs", "Merged", "stage1_merged")),
    "stage2_adapter": os.path.abspath(os.path.join(os.getcwd(), "project_outputs", "Adapter", "instruction_hr_policy_adapter")),
    "merged_stage2": os.path.abspath(os.path.join(os.getcwd(), "project_outputs", "Merged", "stage2_merged")),
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "dtype": None,
    "fix_tokenizer": False,
    "trust_remote_code": True,
    "merge_stage1": True,
    "merge_stage2": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 35,
    "learning_rate": 2e-4,
    "fp16": not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else True,
    "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    "save_strategy": "no",
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config.update(loaded)
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    os.makedirs(config["project_dir"], exist_ok=True)
    for key in ["stage1_adapter", "merged_stage1", "stage2_adapter", "merged_stage2"]:
        os.makedirs(config[key], exist_ok=True)


def load_training_dataset(data_path: str):
    return load_dataset("json", data_files=data_path, split="train")


def format_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    alpaca_prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately answers the HR-related question as the company's HR Policy Assistant.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    alpaca_prompt_template_no_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately answers the HR-related question as the company's HR Policy Assistant.

### Instruction:
{}

### Response:
{}"""

    eos_token = tokenizer.eos_token
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text and str(input_text).strip():
        text = alpaca_prompt_template_with_input.format(instruction, input_text, output) + eos_token
    else:
        text = alpaca_prompt_template_no_input.format(instruction, output) + eos_token
    return {"text": text}


def process_dataset(dataset, tokenizer):
    return dataset.map(lambda example: format_example(example, tokenizer))


def load_stage1_model(config: Dict[str, Any]):
    print("[+] Loading Stage 1 adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["stage1_adapter"],
        max_seq_length=config["max_seq_length"],
        dtype=config["dtype"],
        load_in_4bit=config["load_in_4bit"],
        fix_tokenizer=config["fix_tokenizer"],
        trust_remote_code=config["trust_remote_code"],
    )
    return model, tokenizer


def merge_stage1(model, tokenizer, config: Dict[str, Any]) -> None:
    if not config.get("merge_stage1", True):
        return
    print("[+] Merging Stage 1 adapter...")
    model.save_pretrained_merged(
        config["merged_stage1"],
        tokenizer,
        save_method="merged_16bit",
    )


def reload_merged_model(config: Dict[str, Any]):
    print("[+] Reloading merged Stage 1 model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["merged_stage1"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    return model, tokenizer


def apply_lora(model):
    print("[+] Applying fresh LoRA...")
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def train_sft(config: Dict[str, Any]):
    ensure_directories(config)

    print("[+] Loading dataset...")
    dataset = load_training_dataset(config["data_path"])

    print("[+] Loading Stage 1 model...")
    stage1_model, tokenizer = load_stage1_model(config)
    merge_stage1(stage1_model, tokenizer, config)

    print("[+] Reloading merged model...")
    model, tokenizer = reload_merged_model(config)
    model = apply_lora(model)

    print("[+] Formatting dataset...")
    processed_dataset = process_dataset(dataset, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        args=SFTConfig(
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            learning_rate=config["learning_rate"],
            fp16=config["fp16"],
            bf16=config["bf16"],
            logging_steps=config["logging_steps"],
            optim=config["optim"],
            weight_decay=config["weight_decay"],
            lr_scheduler_type=config["lr_scheduler_type"],
            seed=config["seed"],
            output_dir=config["output_dir"],
            save_strategy=config["save_strategy"],
        ),
    )

    print("[*] Starting fine-tuning...")
    trainer.train()

    print("[+] Saving Stage 2 adapter...")
    model.save_pretrained(config["stage2_adapter"])
    tokenizer.save_pretrained(config["stage2_adapter"])

    if config.get("merge_stage2", True):
        print("[+] Merging Stage 2 model...")
        model.save_pretrained_merged(
            config["merged_stage2"],
            tokenizer,
            save_method="merged_16bit",
        )

    return {
        "status": "success",
        "stage2_adapter": config["stage2_adapter"],
        "merged_stage2": config["merged_stage2"],
    }


def run_training_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved_config = load_config() if config is None else dict(DEFAULT_CONFIG, **config)
    if config is not None:
        resolved_config.update(config)
    return train_sft(resolved_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an instruction fine-tuning model from a notebook workflow")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--stage1-adapter", type=str, default=None)
    parser.add_argument("--merged-stage1", type=str, default=None)
    parser.add_argument("--stage2-adapter", type=str, default=None)
    parser.add_argument("--merged-stage2", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.project_dir:
        config["project_dir"] = args.project_dir
    if args.data_path:
        config["data_path"] = args.data_path
    if args.stage1_adapter:
        config["stage1_adapter"] = args.stage1_adapter
    if args.merged_stage1:
        config["merged_stage1"] = args.merged_stage1
    if args.stage2_adapter:
        config["stage2_adapter"] = args.stage2_adapter
    if args.merged_stage2:
        config["merged_stage2"] = args.merged_stage2

    result = run_training_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

