---
name: aimlite
description: >-
  Expert guide for AIMLite (The Django for AI & Machine Learning).
  Use whenever scaffolding, writing, training, evaluating, or serving
  classical ML models, enterprise RAG knowledge bases, or LoRA/PEFT
  fine-tuning adapters using the aimlite CLI and Python SDK.
---

# AIMLite Skill Guide

This skill teaches coding agents how to build, train, evaluate, and serve production-grade machine learning applications using **AIMLite (v2.1.2)**.

AIMLite operates on **convention over configuration** with **zero-path CLI execution**. Like Django, you organize code into standard modules (`data.py`, `model.py`, `trainer.py`, `evaluator.py`, `inference.py`), and the framework handles discovery, environment resolution, and execution automatically.

---

## 1. Zero-Path CLI Commands

Coding agents should use the `aimlite` CLI commands directly via shell tools:

```bash
# Initialize a new project layout in current directory or named folder
aimlite init <project_name | .> [--type <scratch|rag|adapter>] [--clean] [--install] [-y]

# Synchronize dependencies with .venv from aimlite.json
aimlite install [package_name ...]

# Validate dataset schema, file formats, and partition readiness
aimlite data validate

# Train a model class (auto-discovers class in model.py if omitted)
aimlite train [ModelName]

# Evaluate trained checkpoint against validation/test splits
aimlite evaluate [ModelName]

# Launch multi-model inference server & adaptive web app
aimlite serve [ModelName] [--port 8000] [--host 127.0.0.1] [--api] [--frontend <dir>]

# Run environment and accelerator diagnostic check
aimlite doctor

# Run scripts with project root automatically added to PYTHONPATH
aimlite benchmark [experiments/benchmark.py]
```

### Essential CLI Flags
- `--clean`: Scaffolds pristine skeletons without dummy CSVs or sample data (contracts + docstrings only).
- `--api`: Runs the inference server in headless REST JSON mode (disables web UI, ideal for Docker/Kubernetes).
- `--frontend <dir>`: Mounts a custom SPA build (React, Vue, Vite, Next.js static) on the same port with SPA client routing fallback to `index.html`.
- `--type <scratch|rag|adapter>`: Sets project paradigm.

---

## 2. Standard Directory Layout

When creating or modifying an AIMLite project, strictly adhere to this layout:

```
my_project/
├── aimlite.json             # Manifest: project metadata, paradigm, dependencies
├── data/                    # Datasets (CSV, JSON, JSONL, Parquet, TXT, MD)
├── models/                  # Serialized model checkpoints (*.pkl)
├── artifacts/               # Non-weight artifacts (indexes, configs, vocabularies)
├── experiments/             # Experiment logs, benchmark scripts, metric snapshots
├── templates/               # (Optional) Developer HTML overrides (app.html)
├── data.py                  # Dataset subclass & partition contracts
├── model.py                 # Model subclass & predict implementation
├── trainer.py               # BaseTrainer subclass (training loop)
├── evaluator.py             # BaseEvaluator subclass (validation & metrics)
└── inference.py             # BaseInference subclass (serving handler)
```

---

## 3. The 3 Architectural Pillars

### Pillar 1: Data (`aimlite.data.Dataset`)
Manages data ingestion, schema validation, and deterministic partitioning.
```python
from aimlite import Dataset

class UserDataset(Dataset):
    source = "data/users.csv"        # Optional explicit file path

dataset = UserDataset()
records = dataset.load()             # Returns list of dicts

# Deterministic train/validation/test split
train_data, val_data, test_data = dataset.split(
    train=0.8,
    validation=0.1,
    test=0.1,
    seed=42,                         # Always seed for reproducible evaluation
    shuffle=True
)
```

### Pillar 2: Model (`aimlite.models.Model`)
Framework-agnostic base class with built-in pickle serialization into `models/<name>.pkl`.
```python
from aimlite import Model

class ClassifierModel(Model):
    def __init__(self, name="ClassifierModel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = {}

    def predict(self, inputs, **kwargs):
        # inputs can be dict, list of dicts, or feature list
        return {"prediction": 1, "confidence": 0.94}

model = ClassifierModel()
model.save()                         # Saves to models/ClassifierModel.pkl
model.load()                         # Restores from models/ClassifierModel.pkl
```

### Pillar 3: Lifecycle (`aimlite.lifecycle`)
Separates training, evaluation, and inference concerns.
```python
from aimlite import BaseTrainer, BaseEvaluator, BaseInference

class ModelTrainer(BaseTrainer):
    def fit(self, model, dataset, **kwargs):
        train_data, val_data, _ = dataset.split(train=0.8, validation=0.2, test=0.0)
        # Execute real training loop here...
        model.save()
        return {"status": "completed", "final_loss": 0.012}

class ModelEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        _, _, test_data = dataset.split(train=0.8, validation=0.1, test=0.1)
        # Compute accuracy, precision, recall, or RMSE...
        return {"accuracy": 0.96, "test_samples": len(test_data)}

class ModelInference(BaseInference):
    def run(self, model, raw_input, **kwargs):
        # Validate input and dispatch prediction
        return model.predict(raw_input)
```

---

## 4. The 3 AI Paradigms

### Paradigm 1: Scratch Training (Classical & Deep ML)
Used for Scikit-learn, PyTorch, TensorFlow, XGBoost, or pure Python algorithms.
- **Scaffold**: `aimlite init my_model --type scratch`
- **Serving UI**: Automatically generates an interactive Feature Form with auto-discovered columns, preset buttons, and JSON mode toggle.

### Paradigm 2: Enterprise RAG (`aimlite.rag`)
Production knowledge retrieval with multi-provider LLM synthesis and vector indexing.
- **Scaffold**: `aimlite init my_rag --type rag`
- **Core Components**:
  - `Document(content, metadata, id)`: Container for indexed content.
  - `SmartChunker(max_chunk_size=600, chunk_overlap=60)`: Hierarchical document chunker decomposing along Markdown headers (`#`, `##`), paragraphs, sentence boundaries, word tokens, and character windows.
  - `MemoryVectorStore(embedding_fn)`: In-memory cosine similarity store.
  - `PostgresVectorStore(connection_string)`: PostgreSQL `pgvector` store with ORM migration support.
  - `TfidfEmbedding()`: Zero-dependency pure Python TF-IDF embedding.
  - `SentenceTransformerEmbedding(model_name)`: Dense semantic embeddings (requires `sentence-transformers`).
  - `BaseChatProvider`: Supported providers:
    - `OpenAIChatProvider(api_key=..., model="gpt-4o")`
    - `AnthropicChatProvider(api_key=..., model="claude-3-5-sonnet-20241022")`
    - `GeminiChatProvider(api_key=..., model="gemini-1.5-flash")`
    - `OllamaChatProvider(model="llama3.2")`
    - `LocalChatProvider(model_or_fn=...)`
    - `MockChatProvider(fixed_response=...)` (for test harnesses only)
  - `KnowledgeModel`: Coordinates retriever, chunker, and provider. **Must be initialized with an explicit `chat_provider` in production.**
  - `RAGTrainer`: Ingests dataset documents, generates chunks, and builds persistent vector index.

```python
from aimlite.rag import (
    KnowledgeModel,
    SmartChunker,
    TfidfEmbedding,
    MemoryVectorStore,
    OpenAIChatProvider
)

chat_provider = OpenAIChatProvider(api_key="sk-...")
model = KnowledgeModel(
    name="SupportRAG",
    chat_provider=chat_provider,
    embedding_fn=TfidfEmbedding(),
    chunker=SmartChunker(max_chunk_size=500, chunk_overlap=50)
)
model.add_document("AIMLite enables zero-path machine learning.", title="Overview")
response = model.predict("What is AIMLite?")
# Returns: {"answer": "...", "sources": [{"content": "...", "score": 0.95}]}
```

### Paradigm 3: LoRA & PEFT Fine-Tuning (`aimlite.adapters`)
Parameter-efficient fine-tuning via low-rank matrix decomposition:
$$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- **Scaffold**: `aimlite init my_lora --type adapters`
- **Core Components**:
  - `LoRALayer(in_features, out_features, r=8, lora_alpha=16)`: Low-rank linear layer with forward ($h = W_0 x + \frac{\alpha}{r} B A x$) and analytical backward updates (`backward_B`, `backward_A`).
  - `AdapterModel(base_model, adapter_config)`: Base adapter wrapper with `merge_weights()`, `unmerge_weights()`, and `get_trainable_parameters()`.
  - `AdapterTrainer`: Executes true gradient backpropagation on LoRA matrices $A$ and $B$ with MSE loss and gradient clipping ($[-1.0, 1.0]$).
  - `MultiAdapterManager`: Hot-swaps multiple adapter weights on top of a shared frozen base model without reloading.
  - `Hugging Face PEFT compatibility`: Generates standard `adapter_config.json` and `adapter_model.pkl`.

---

## 5. Multi-Model Serving & Endpoints

When `aimlite serve` is executed, it starts a threaded HTTP server (`ThreadingHTTPServer`) with non-blocking concurrent request handling:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` or `/app` | Paradigm-adaptive Web Application UI with live Model Switcher |
| `GET` | `/chat` | Dedicated Conversational Chat Playground (with Markdown response rendering) |
| `GET` | `/models` | JSON inventory of all registered models and metadata |
| `GET` | `/health` | Server health, active model, and uptime status |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/openapi.json` | Full OpenAPI 3.0 specification schema |
| `POST` | `/predict` | Primary inference dispatch (accepts optional `"model": "<name>"`) |
| `POST` | `/models/{name}/predict` | Direct inference target for a specific registered model |
| `POST` | `/models/{name}/search` | Semantic search endpoint for RAG models (retrieval only, no LLM) |

---

## 6. Golden Rules for Coding Agents

1. **Zero Path Boilerplate**: Place raw files in `data/`, checkpoints in `models/`, and artifacts in `artifacts/`. Never hardcode absolute paths or write `os.path.join(__file__, '..', '..')`.
2. **Never Mock in Production Paths**: Never inject `MockChatProvider` as the default in production code. Require explicit provider configuration via `.env` or constructor arguments.
3. **No Silent Fallbacks**: If an optional library (`sentence-transformers`, `torch`, `peft`, `psycopg2`) is missing, raise an explicit `ImportError` explaining how to install it. Never silently degrade to an inferior implementation.
4. **Deterministic Partitioning**: Always specify `seed` in `Dataset.split()` so experiments and benchmarks are 100% reproducible.
5. **No Fake Training Results**: Custom trainers must compute real forward passes, real losses, and measurable parameter updates.
6. **Threaded Concurrency**: If custom HTTP request handling is added, always ensure it is non-blocking (`ThreadingMixIn` or async) so concurrent predictions or health checks are not blocked.
7. **Clean Code**: Do not leave `TODO` or `FIXME` comments in generated code.
