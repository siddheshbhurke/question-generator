# ✅ FASTAPI BACKEND TO SERVE MICROSOFT PHI-2 MODEL FOR QUESTION GENERATION

# Step 1: Install dependencies (in terminal or requirements.txt)
# pip install fastapi uvicorn transformers torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Request body model
class InputData(BaseModel):
    use_case: str
    context: str

# Load lightweight instruction-following model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Check for GPU availability (optional)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()

# Prompt builder
def build_prompt(use_case: str, context: str) -> str:
    return f"""
Generate exactly 10 survey questions for the following use case: \"{use_case}\"

Context provided by user: \"{context}\"

Requirements:
- Mix question types: multiple choice (with 3–5 options), yes/no, range-based (1–5), and open-ended
- Make questions specific to the context
- Return JSON array only
- Do not return any explanation or formatting beyond the JSON output
"""

@app.post("/generate")
async def generate_questions(data: InputData):
    prompt = build_prompt(data.use_case, data.context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"questions": response}

# To run locally:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
