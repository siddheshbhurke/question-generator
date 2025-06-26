# ✅ FASTAPI BACKEND TO SERVE FAST SURVEY QUESTION GENERATION USING FLAN-T5-SMALL WITH QUESTION TYPES AND OPTIONS

# Step 1: Install dependencies (run in Kaggle cell)
# !pip install fastapi uvicorn transformers

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

# Define request body schema
class InputData(BaseModel):
    use_case: str
    context: str

# Use a fast and light model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()

# Prompt builder for different question types
def build_typed_prompt(use_case: str, context: str, qtype: str, idx: int) -> str:
    if qtype == "multiple_choice":
        return (
            f"Generate a multiple choice survey question based on the following:\n"
            f"- Use Case: {use_case}\n"
            f"- Context: {context}\n"
            f"Return a JSON object with keys: 'question', 'type', and 'options' (3-5 relevant choices)."
        )
    elif qtype == "yes_no":
        return (
            f"Generate a yes/no survey question.\n"
            f"Use Case: {use_case}\nContext: {context}\n"
            f"Return only the question."
        )
    elif qtype == "range":
        return (
            f"Generate a 1–5 scale rating question (Very Poor to Excellent).\n"
            f"Use Case: {use_case}\nContext: {context}\n"
            f"Return only the question."
        )
    elif qtype == "open_ended":
        return (
            f"Generate an open-ended survey question.\n"
            f"Use Case: {use_case}\nContext: {context}\n"
            f"Return only the question."
        )

# Inference logic for a single question
def generate_typed_question(use_case, context, qtype, idx):
    prompt = build_typed_prompt(use_case, context, qtype, idx)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    if qtype == "multiple_choice":
        try:
            raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            parsed = json.loads(raw)
            parsed["type"] = qtype
            return parsed
        except Exception:
            return {"question": "Failed to generate question.", "type": qtype, "options": []}
    elif qtype == "range":
        return {
            "question": tokenizer.decode(output[0], skip_special_tokens=True).strip(),
            "type": qtype,
            "minValue": 1,
            "maxValue": 5,
            "minLabel": "Very Poor",
            "maxLabel": "Excellent"
        }
    else:
        return {
            "question": tokenizer.decode(output[0], skip_special_tokens=True).strip(),
            "type": qtype
        }

@app.post("/generate")
async def generate_questions(data: InputData):
    qtypes = ["multiple_choice", "yes_no", "range", "open_ended"] * 3
    results = [generate_typed_question(data.use_case, data.context, qt, i) for i, qt in enumerate(qtypes[:10])]
    return {"questions": results}




