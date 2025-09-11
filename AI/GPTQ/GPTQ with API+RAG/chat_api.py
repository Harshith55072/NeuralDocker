# ====================================
# Imports and Dependencies
# ====================================
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uvicorn

# ====================================
# Model & Knowledge Base Configuration
# ====================================
# Replace with your own local path or Hugging Face model name
MODEL_PATH = "path_to_your_model"

# Path to your knowledge base file (.json or .txt)
DATA_PATH = "path_to_your_data.json"

# ====================================
# Load Tokenizer & Model
# ====================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",        # auto-select GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True
)
model.config.pad_token_id = model.config.eos_token_id

# ====================================
# Custom Stopping Criteria
# ====================================
class StopOnTokens(StoppingCriteria):
    """Stop generation when one of the given token IDs appears."""
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0][-1] == stop_id for stop_id in self.stop_token_ids)

stop_ids = [model.config.eos_token_id]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

# ====================================
# Role-based Prompt Templates
# ====================================
roles = {
    "default": "### Instruction:\n{context}\n{user}\n\n### Response:",
    "customer_service": (
        "### Instruction:\nYou are a customer service AI. Be polite and concise.\n\n{context}\n{user}\n\n### Response:"
    ),
    "customer_service2": (
        "### Instruction:\nYou are a customer service AI. Be polite, professional, and brief (2–3 sentences).\n\n{context}\n{user}\n\n### Response:"
    ),
    "tech_support": (
        "### Instruction:\nYou are a technical support AI. Provide clear, step-by-step help.\n\n{context}\n{user}\n\n### Response:"
    ),
    "friendly_chat": (
        "### Instruction:\nYou are a friendly chatbot. Keep things casual and engaging.\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor": (
        "### Instruction:\nYou are a friendly mentor who helps software engineering students. "
        "Answer their doubts and guide them. Don’t assume anything unless asked, and don’t mention fields unless requested. "
        "Keep responses short and to the point. Ask clarification questions only if needed.\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor-v2": (
        "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. "
        "Do not assume the field or topic. Keep answers short and clear. Ask questions only if you need clarification.\n\n{context}\n{user}\n\n### Response:"
    )
}
active_role = "Career_mentor"  # default role

# ====================================
# Load Knowledge Base (RAG)
# ====================================
print("📚 Loading knowledge base...")

docs = []
if DATA_PATH.endswith(".json"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            docs = list(data.values())
        elif isinstance(data, list):
            docs = data
elif DATA_PATH.endswith(".txt"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

# Build embeddings + FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ====================================
# FastAPI App
# ====================================
app = FastAPI(title="GPTQ + RAG Chat API")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role  # optional, defaults to Career_mentor

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_input = request.user_input
    role = request.role if request.role in roles else active_role

    # Skip retrieval for greetings or very short queries
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""
    else:
        # Retrieve top-3 relevant docs
        query_vec = embedder.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)

        retrieved_docs = []
        for i in I[0]:
            if isinstance(docs[i], dict) and "text" in docs[i]:
                retrieved_docs.append(docs[i]["text"])
            else:
                retrieved_docs.append(str(docs[i]))
        context = "\n".join(retrieved_docs)

    # Build final prompt
    full_prompt = roles[role].format(context=context, user=user_input)

    # Generate response
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        stopping_criteria=stopping_criteria,
    )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract only model's answer after "### Response:"
    if "### Response:" in output_text:
        response = output_text.split("### Response:")[-1].strip()
    else:
        response = output_text.strip()

    return {"response": response}

# ====================================
# Run Server
# ====================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
