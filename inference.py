"""
inference.py

Inference module for the HR Policy Assistant.
Loads the fine-tuned model once and provides a reusable
generate_response() function.
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # ------------------------------------------------------
    # FOR LOCAL TESTING
    # ------------------------------------------------------
    "model_path": os.path.join(
        BASE_DIR,
        "project_outputs",
        "Merged",
        "stage2_merged",
    ),

    # ------------------------------------------------------
    # AFTER UPLOADING TO HUGGING FACE
    # Replace the above with:
    #
    # "model_path": "KHANmdAFFAN/hr-policy-assistant-model",
    #
    # ------------------------------------------------------

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

def load_model(
    model_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase]:

    path = model_path or CONFIG["model_path"]

    logger.info("Loading model: %s", path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    FastLanguageModel.for_inference(model)

    logger.info("Model loaded successfully.")

    return model, tokenizer

# ==========================================================
# Load Once
# ==========================================================

MODEL, TOKENIZER = load_model()

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

    prompt = build_prompt(question, input_text)

    device = next(MODEL.parameters()).device

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens or CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            do_sample=False,
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
# Command Line Interface
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="HR Policy Assistant Inference",
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="HR-related question",
    )

    parser.add_argument(
        "--input-text",
        type=str,
        default="",
        help="Optional context",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=CONFIG["max_new_tokens"],
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

    print("\n" + "=" * 60)
    print("HR Policy Assistant")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
