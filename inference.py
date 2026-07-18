"""
inference.py

Inference module for the HR Policy Assistant.
Loads the Hugging Face model once and provides
generate_response() for Gradio or CLI usage.
"""

# ==========================================================
# IMPORTANT
# ==========================================================

import unsloth

import argparse
import logging
from typing import Optional

import torch
from unsloth import FastLanguageModel

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

CONFIG = {
    "model_name": "Toji619/hr-policy-assistant-model",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "max_new_tokens": 200,
    "temperature": 0.2,
    "top_p": 0.9,
}

# ==========================================================
# Prompt Templates
# ==========================================================

PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately answers the HR-related question as the company's HR Policy Assistant.

### Instruction:
{}

### Input:
{}

### Response:
"""

PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately answers the HR-related question as the company's HR Policy Assistant.

### Instruction:
{}

### Response:
"""

# ==========================================================
# Load Model
# ==========================================================

logger.info("Loading model...")

MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=CONFIG["load_in_4bit"],
)

FastLanguageModel.for_inference(MODEL)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device : {DEVICE}")

# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(
    question: str,
    input_text: str = "",
) -> str:

    if input_text.strip():

        return PROMPT_WITH_INPUT.format(
            question,
            input_text,
        )

    return PROMPT_NO_INPUT.format(question)

# ==========================================================
# Generate Response
# ==========================================================

def generate_response(
    question: str,
    input_text: str = "",
    max_new_tokens: Optional[int] = None,
) -> str:

    prompt = build_prompt(
        question,
        input_text,
    )

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens or CONFIG["max_new_tokens"],
            do_sample=False,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            use_cache=True,
        )

    decoded = TOKENIZER.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[-1]

    return decoded.strip()

# ==========================================================
# CLI
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--input-text",
        default="",
        type=str,
    )

    parser.add_argument(
        "--max-new-tokens",
        default=200,
        type=int,
    )

    return parser.parse_args()

# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    response = generate_response(
        question=args.question,
        input_text=args.input_text,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n")
    print("=" * 70)
    print("HR Policy Assistant")
    print("=" * 70)
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    main()
