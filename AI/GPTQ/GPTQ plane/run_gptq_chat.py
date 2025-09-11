# =========================
# Imports and Dependencies
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# ===================
# Model Configuration
# ===================
MODEL_PATH = r"E:\text-generation-webui-main\text-generation-webui-main\user_data\models\TheBloke_MythoMist-7B-GPTQ"

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

# Set padding token to the same as EOS (end-of-sequence) token
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

# List of token IDs that will stop generation
stop_ids = [model.config.eos_token_id]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

# ==========================
# Role-based Prompt Templates
# ==========================
roles = {
    "default": "### Instruction:\n{user}\n\n### Response:",
    "customer_service": (
        "### Instruction:\nYou are a customer service AI. Be polite and concise and keep it short.\n\n{user}\n\n### Response:"
    ),
    "customer_service2": (
        "### Instruction:\nYou are a customer service AI. Be polite, professional, concise, and brief. Limit your response to 2-3 sentences.\n\n{user}\n\n### Response:"
    ),
    "tech_support": (
        "### Instruction:\nYou are a technical support AI. Provide clear, step-by-step help.\n\n{user}\n\n### Response:"
    ),
    "friendly_chat": (
        "### Instruction:\nYou are a friendly chatbot. Keep things casual and engaging.\n\n{user}\n\n### Response:"
    )
}

# =================
# Active Chat Role
# =================
active_role = "customer_service"  # Change this as needed

# ================
# Interactive Chat
# ================
print("🔹 GPTQ Chat Ready (streaming, type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break

    # Build prompt using the selected role
    full_prompt = roles[active_role].format(user=user_input)

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate model response (streamed to console)
    model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        stopping_criteria=stopping_criteria,
        streamer=streamer  # streamed output – nothing returned
    )
