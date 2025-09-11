# =========================
# Imports and Dependencies
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# ===================
# Model Configuration
# ===================
MODEL_PATH = r"E:\text-generation-webui-main\text-generation-webui-main\user_data\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"

# Location of your knowledge base file (can be .txt or .json)
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

# =======================
# Text Streaming Settings
# =======================
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

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
    "customer_service": (
        "### Instruction:\nYou are a customer service AI. Be polite and concise.\n\n{context}\n{user}\n\n### Response:"
    ),
    "customer_service2": (
        "### Instruction:\nYou are a customer service AI. Be polite, professional, concise, and brief. Limit your response to 2-3 sentences.\n\n{context}\n{user}\n\n### Response:"
    ),
    "tech_support": (
        "### Instruction:\nYou are a technical support AI. Provide clear, step-by-step help.\n\n{context}\n{user}\n\n### Response:"
    ),
    "friendly_chat": (
        "### Instruction:\nYou are a friendly chatbot. Keep things casual and engaging.\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor": (
        "### Instruction:\nYou are a friendly mentor.Who helps softwear enginnering students, answer there doubts and guide them. dont assume anything until they ask and dont mention any fields until they ask,"
        " try asking a quetion at the end for your information if needed. try keeping it short and to the point\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor-v2": (
    "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. Do not assume the field or topic. Keep answers short and clear. Ask questions only if you need clarification.\n\n{context}\n{user}\n\n### Response:"
)

}

active_role = "Career_mentor"  # Change this as needed

# ==========================
# Load Knowledge Base (RAG)
# ==========================
print("📚 Loading knowledge base...")

# Load documents from .json or .txt
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

# Initialize embedding model & FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ================
# Interactive Chat
# ================
print("🔹 GPTQ + RAG Chat Ready (type 'exit' to quit)")

# Define greetings for shortcut
greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break

    # --- Check if input is a simple greeting ---
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""  # Skip RAG for greetings
    else:
        # === RAG Retrieval ===
        query_vec = embedder.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)  # top-3 results
        # Extract only the "text" field for context
        retrieved_docs = [docs[i]["text"] for i in I[0]]
        context = "\n".join(retrieved_docs)

    # Build prompt with role + retrieved context
    full_prompt = roles[active_role].format(context=context, user=user_input)

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate model response (streamed to console)
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
