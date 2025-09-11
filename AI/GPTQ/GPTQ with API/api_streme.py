# =========================
# Imports and Dependencies
# =========================
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch
import uvicorn
import threading

# ===================
# Model Configuration
# ===================
MODEL_PATH = "path_to_your_model"

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
# Role-based Prompt Template
# ==========================
roles = {
    "default": "### Instruction:\n{user}\n\n### Response:",
    "friendly_chat": (
        "### Instruction:\nYou are a friendly chatbot. Keep things casual and engaging.\n\n{user}\n\n### Response:"
    ),
    "career_mentor": (
        "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. "
        "Do not assume the field or topic. Keep answers short and clear.\n\n{user}\n\n### Response:"
    ),
}
active_role = "friendly_chat"

# ================
# FastAPI App
# ================
app = FastAPI(title="GPTQ Streaming API")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    user_input = request.user_input
    role = request.role if request.role in roles else active_role

    # Build prompt
    full_prompt = roles[role].format(user=user_input)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Streamer setup
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

    # Token generator for SSE-like streaming
    def token_generator():
        for token in streamer:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
