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

# ====================================
# Model Configuration
# ====================================
# Replace with your downloaded or Hugging Face model path/name
MODEL_PATH = "path_to_your_model" 

# ====================================
# Load Tokenizer & Model
# ====================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True   # ensures it only loads from local files
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",       # automatically selects available GPU/CPU
    trust_remote_code=True,  # allows loading models with custom code
    torch_dtype=torch.float16,
    local_files_only=True
)

# Use EOS (end-of-sequence) token as the padding token
model.config.pad_token_id = model.config.eos_token_id

# ====================================
# Text Streaming (live output display)
# ====================================
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,        # don’t repeat user input in output
    skip_special_tokens=True # hide tokens like <EOS>, <PAD>, etc.
)

# ====================================
# Custom Stopping Criteria
# ====================================
class StopOnTokens(StoppingCriteria):
    """Stops generation when a specified token ID is encountered."""
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0][-1] == stop_id for stop_id in self.stop_token_ids)

# List of tokens that will stop generation
stop_ids = [model.config.eos_token_id]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

# ====================================
# Role-based Prompt Templates
# ====================================
roles = {
    "default": (
        "### Instruction:\n{user}\n\n### Response:"
    ),
    "customer_service": (
        "### Instruction:\nYou are a customer service AI. "
        "Be polite and concise.\n\n{user}\n\n### Response:"
    ),
    "customer_service2": (
        "### Instruction:\nYou are a customer service AI. "
        "Be polite, professional, and brief. Limit response to 2–3 sentences.\n\n{user}\n\n### Response:"
    ),
    "tech_support": (
        "### Instruction:\nYou are a technical support AI. "
        "Provide clear, step-by-step help.\n\n{user}\n\n### Response:"
    ),
    "friendly_chat": (
        "### Instruction:\nYou are a friendly chatbot. "
        "Keep things casual and engaging.\n\n{user}\n\n### Response:"
    )
}

# ====================================
# Active Role (change as needed)
# ====================================
active_role = "customer_service"

# ====================================
# Interactive Chat Loop
# ====================================
print("🤖 Chat Ready (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Exiting chat.")
        break

    # Build full prompt for the model using active role
    full_prompt = roles[active_role].format(user=user_input)

    # Tokenize input and move tensors to the same device as the model
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate response (streamed to console in real-time)
    model.generate(
        **inputs,
        max_new_tokens=100,       # cap on output length
        do_sample=True,           # sampling for varied responses
        temperature=0.7,          # creativity level
        top_p=0.9,                # nucleus sampling
        top_k=50,                 # limit to top-k tokens
        repetition_penalty=1.2,   # discourage repeated phrases
        stopping_criteria=stopping_criteria,
        streamer=streamer         # stream output directly to console
    )
