# =========================
# Imports and Dependencies
# =========================
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uvicorn
import threading

# ===================
# Model Configuration
# ===================
MODEL_PATH = r"E:\text-generation-webui-main\text-generation-webui-main\user_data\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"
DATA_PATH = r"C:\Users\Lenovo\Documents\programing\data\software_career_knowledge.json"  

# ======================
# Load Tokenizer & Model
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True
)
model.config.pad_token_id = model.config.eos_token_id

# ============================
# Custom Stopping Criteria
# ============================
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0][-1] == stop_id for stop_id in self.stop_token_ids)

stop_ids = [model.config.eos_token_id]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

# ==========================
# Role-based Prompt Templates
# ==========================
roles = {
    "default": "### Instruction:\n{context}\n{user}\n\n### Response:",
    "Career_mentor": (
        "### Instruction:\nYou are a friendly mentor.Who helps softwear enginnering students, answer there doubts and guide them. dont assume anything until they ask and dont mention any fields until they ask,"
        " try asking a quetion at the end for your information if needed. try keeping it short and to the point\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor-v2": (
        "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. Do not assume the field or topic. Keep answers short and clear. Ask questions only if you need clarification.\n\n{context}\n{user}\n\n### Response:"
    )
}
active_role = "Career_mentor"  # default role

# ==========================
# Load Knowledge Base (RAG)
# ==========================
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

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ================
# FastAPI App
# ================
app = FastAPI(title="GPTQ + RAG Chat API")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role  # optional, defaults to Career_mentor

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    user_input = request.user_input
    role = request.role if request.role in roles else active_role

    # RAG context
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""
    else:
        query_vec = embedder.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)
        retrieved_docs = []
        for i in I[0]:
            if isinstance(docs[i], dict) and "text" in docs[i]:
                retrieved_docs.append(docs[i]["text"])
            else:
                retrieved_docs.append(str(docs[i]))
        context = "\n".join(retrieved_docs)

    # Build prompt
    full_prompt = roles[role].format(context=context, user=user_input)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Create a streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Background thread for generation
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Generator for StreamingResponse
    def token_generator():
        for token in streamer:
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
