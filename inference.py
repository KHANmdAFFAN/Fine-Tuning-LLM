%%writefile inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Toji619/hr-policy-assistant-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model.to(DEVICE)
model.eval()


def build_prompt(question, input_text=""):
    if input_text.strip():
        return f"""Below is an instruction that describes a task.

### Instruction:
{question}

### Input:
{input_text}

### Response:
"""
    return f"""Below is an instruction that describes a task.

### Instruction:
{question}

### Response:
"""


def generate_response(question, input_text=""):
    prompt = build_prompt(question, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.2,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    return text.strip()
