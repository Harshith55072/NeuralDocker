# ====================================
# Imports and Dependencies
# ====================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# ====================================
# Model & Data Configuration
# ====================================
# Replace these with your own model path / Hugging Face model name
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
    device_map="auto",        # automatically chooses GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True
)

# Set padding token = EOS token
model.config.pad_token_id = model.config.eos_token_id

# ====================================
# Text Streaming (live output display)
# ====================================
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

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
    "default": (
        "### Instruction:\n{context}\n{user}\n\n### Response:"
    ),
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
        "### Instruction:\nYou are a friendly mentor for software engineering students. "
        "Answer their doubts and guide them. Don’t assume anything until asked, and don’t bring up fields unless requested. "
        "Keep responses short and to the point. Ask a clarification question if needed.\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor-v2": (
        "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. "
        "Do not assume the field or topic. Keep answers short and clear. Ask questions only if you need clarification.\n\n{context}\n{user}\n\n### Response:"
    )
}

# Active role (can be changed on the fly)
active_role = "Career_mentor"

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

# Initialize embedding model + FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ====================================
# Interactive Chat Loop
# ====================================
print("🤖 GPTQ + RAG Chat Ready (type 'exit' to quit)")

# Quick shortcut: greetings that skip retrieval
greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break

    # Skip RAG for greetings or very short inputs
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""
    else:
        # Retrieve top-3 most relevant docs
        query_vec = embedder.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)

        # If docs are dicts with "text" fields, extract those
        if isinstance(docs[0], dict) and "text" in docs[0]:
            retrieved_docs = [docs[i]["text"] for i in I[0]]
        else:
            retrieved_docs = [docs[i] for i in I[0]]

        context = "\n".join(retrieved_docs)

    # Build final prompt (role + context + user query)
    full_prompt = roles[active_role].format(context=context, user=user_input)

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate response (streamed to console)
    model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        stopping_criteria=stopping_criteria,
        streamer=streamer
    )
