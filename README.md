# NeuralDocker

NeuralDocker is a developer toolkit for running AI models locally. It includes pre-built Python scripts, microservices, and model configurations to help you get started quickly. The project is still in early stages: features like data generation, QLoRA support, and AWQ library integration are coming soon.

---

## Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Version / Model Variants](#version--model-variants)  
- [Dependencies](#dependencies)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Downloading a model](#downloading-a-model)  
  - [Running locally](#running-locally)  
  - [Running with RAG / API / RAG + API](#running-with-rag--api--rag--api)  
- [Roadmap](#roadmap)  
- [Contributing](#contributing)  
- [License](#license)

---

## Features

- Pre-built scripts to run AI models locally  
- Microservices ready to expose model inference APIs  
- Support for different model configurations (plain GPTQ, with RAG, with API, etc.)  
- Modular architecture; components under development for data generation, QLoRA, AWQ support  

---

## Project Structure

Here’s an overview of the folders and what they contain:

```
NeuralDocker/
├── AI/              # Scripts and code for model inference and wrapper logic
├── QLoRa/           # Placeholder for future QLoRA-based modules
├── DATA Generation/  # Data generation utilities (coming soon)
├── …                # Other microservices, helpers, configs
```

---

## Version / Model Variants

NeuralDocker includes several “version planes” (variants) so you can pick based on your needs:

| Variant | What you get | When to use it |
|---|---|---|
| **Plain GPTQ model** | Just the model loaded locally with inference scripts. No RAG (Retrieval-Augmented Generation), no API server. | For offline use, experiments, or integration into other systems. |
| **GPTQ + RAG** | Model + a RAG module (you need a database or vector store), enabling you to augment the model’s answers based on external documents. | When you want improved factuality / knowledge that’s beyond what the base model knows. |
| **API** | A microservice/API wrapper around the model so you can send requests (HTTP etc.). | If you want to deploy or access the model via network / make apps consume it. |
| **RAG + API** | The full setup: model + RAG module + API wrapper. | Best for production‐like use, or when serving many users with knowledge augmented generation. |

You can choose whichever variant suits your use case. The API or RAG parts may have additional configuration steps.

---

## Dependencies

Here are the libraries / tools you’ll need. These are approximate; always check the specific script’s imports for updates.

- Python (3.8+)  
- PyTorch  
- Transformers (Hugging Face)  
- Accelerate (for faster model loading if supported)  
- FAISS / some vector store (if using RAG)  
- Flask / FastAPI / or similar (for API wrapper)  
- Other utilities: tokenizers, numpy, etc.

---

## Installation

Below are steps to set up NeuralDocker on your machine.

```bash
# 1. Clone the repo
git clone https://github.com/Harshith55072/NeuralDocker.git
cd NeuralDocker

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

If there is no `requirements.txt`, you can manually install via:

```bash
pip install torch transformers accelerate flask fastapi faiss-cpu
# plus any other libraries you see imported under AI/, etc.
```

---

## Usage

Here’s how to use the scripts / microservices.

### Downloading a Model

You’ll need to download a GPTQ model (or any compatible model). A recommended source is the Hugging Face Hub (for example “blokes” models).

1. Go to Hugging Face → choose a GPTQ model.  
2. Download the model to your local machine (you’ll get a folder with model weights and config).  
3. Note the path/location where you stored it.

### Running Locally (Plain GPTQ)

Suppose you have downloaded the model and placed it at `~/models/my‐gptq‐model/`.

Then:

```bash
# Example: run the script
python AI/run_model.py --model_path ~/models/my-gptq-model/
```

Adjust flags/arguments as per the script. Look for args like `--model_path`.

### Running with RAG

If you want to enable RAG, you’ll need a knowledge base / vector store, e.g. documents stored, embedded, and index built (FAISS or similar).

```bash
# Example:
python AI/run_model_rag.py --model_path ~/models/my-gptq-model/ --rag_index_path ~/data/my_rag_index/ --docs_path ~/docs/
```

Make sure your rag index or vector store is built (there may be helper scripts in the `DATA Generation` folder).

### Running as API / Microservice

To serve model inference (with or without RAG) via API:

```bash
# Example:
python API/app.py --model_path ~/models/my-gptq-model/ --rag_index_path ~/data/my_rag_index/
```

This should start a web server (check which framework: Flask / FastAPI) on some port (e.g. `localhost:8000`), so you can send HTTP requests:

```http
POST /predict
{
   "prompt": "Your input prompt here"
}
```

Adjust endpoints per the API’s code.

---

## Roadmap

What’s coming / planned in upcoming versions:

- Full **Data Generation** tools  
- **QLoRA**‐based training and inference  
- Support for **AWQ** library (for quantization / performance)  
- Better documentation, examples, and maybe GUI / Web UI  
- More model variants, better tooling around index building for RAG  

---

## Contributing

You’re welcome to contribute! Some suggestions:

- Add tests for model loading / inference  
- Improve error handling, especially around invalid model paths or missing RAG assets  
- Add examples / sample configs  
- Add Docker setup if desired  
- Improve performance / add memory / GPU optimisations  

---

## License

*(Choose your license here, e.g. MIT, Apache 2.0, etc.)*
