# 📖 Project Overview

Wikipedia Assistant is an intelligent Q&A system based on Retrieval-Augmented Generation (RAG). It integrates the Wikipedia API, vector databases, and large language models to provide accurate, real-time knowledge answers via a RESTful API.

This project was built with pair programming using Cursor.

## ✨ Features
- **Intelligent Q&A**: Accurate answers based on Wikipedia data
- **Real-time Retrieval**: Fast vector search for relevant information
- **Multilingual Support**: Handles both Chinese and English queries
- **Safety Validation**: Guardrails for jailbreak/toxicity detection
- **RESTful API**: Easy integration for external applications
- **Modular Design**: Clean separation of core, API, and scripts

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **LLM**: Cohere Command
- **Vectorization**: HuggingFace sentence-transformers
- **Reranking**: Cohere Rerank
- **Vector DB**: FAISS
- **Data Source**: Wikipedia API
- **Validation**: Guardrails AI

## 🏗️ System Architecture
```
Wikipedia API → Data Fetch → Text Chunking → Vectorization → FAISS Storage
                                                        ↓
User Query → Vector Retrieval → Rerank → LLM Generation → Guardrails Validation → Response
```

## 📁 Project Structure
```
wiki_search/
├── app/                # Main application
│   ├── api/            # API routes
│   ├── core/           # Core RAG logic
│   └── config.py       # Settings & API keys
├── scripts/            # Utility scripts
├── tests/              # Unit & integration tests
├── data/               # Raw data
├── faiss_index/        # Vector DB
├── logs/               # Log files
├── pyproject.toml      # Poetry config
├── docker-compose.yml  # Docker Compose config
├── Dockerfile          # Docker build config
├── env.example         # Environment variable template
└── README.md           # Project documentation
```

---

# 🏁 Getting Started

## Prerequisites
- Python 3.12+
- Poetry
- FastAPI
- Uvicorn

## Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wiki_search
   ```
2. **Install dependencies**
   ```bash
   poetry install
   ```
3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

## Running the API Server
**Background:**
```cmd
start uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```
**Foreground (blocks terminal):**
```cmd
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Health check:**
```cmd
curl -X GET "http://127.0.0.1:8000/health"
# Should return: {"status":"ok","timestamp":"..."}
```

---

# 📚 API Usage Guide

## Endpoints Overview

| Method | Endpoint           | Description                                 |
|--------|--------------------|---------------------------------------------|
| GET    | /                  | API info                                    |
| GET    | /health            | Health check                                |
| GET    | /docs              | Swagger UI                                  |
| GET    | /redoc             | ReDoc documentation                         |
| POST   | /api/query         | Single RAG query                            |
| POST   | /api/batch-query   | Batch RAG queries                           |
| GET    | /api/status        | System status and statistics                |
| GET    | /api/config        | Get current configuration                   |
| POST   | /api/reset-stats   | Reset API statistics                        |

## Example: Single Query

**curl**
```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who is the current president of Taiwan?"}'
```

**Python (requests)**
```python
import requests
resp = requests.post(
    "http://127.0.0.1:8000/api/query",
    json={"query": "Who is the current president of Taiwan?"}
)
print(resp.json())
```

## Example: Batch Query

**curl**
```bash
curl -X POST "http://127.0.0.1:8000/api/batch-query" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["What is the capital of Taiwan?", "Who are the presidents of Taiwan?"]}'
```

**Python (requests)**
```python
import requests
resp = requests.post(
    "http://127.0.0.1:8000/api/batch-query",
    json={"queries": ["What is the capital of Taiwan?", "Who are the presidents of Taiwan?"]}
)
print(resp.json())
```

## Example: Health Check

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

## Example: System Status

```bash
curl -X GET "http://127.0.0.1:8000/api/status"
```

## Example: Get Configuration

```bash
curl -X GET "http://127.0.0.1:8000/api/config"
```

## Example: Reset Statistics

```bash
curl -X POST "http://127.0.0.1:8000/api/reset-stats"
```

---

# ❓ FAQ & Error Handling

### Q: What should I do if I get a 400 Bad Request?
A: Check that your JSON body matches the required format. For example, for `/api/query`, you must provide `{ "query": "..." }`.

### Q: What does a 500 Internal Server Error mean?
A: This indicates a server-side error. Check the error message in the response (if available) and ensure your input is valid. If the problem persists, check the server logs or open an issue.

### Q: How do I get/set my API key?
A: Set your Cohere API key in the `.env` file as `COHERE_API_KEY=your_key_here`.

### Q: What if I hit a rate limit or external API quota?
A: You may need to wait and try again later, or check your Cohere account for quota status.

### Q: How do I run tests?
A: Use `poetry run pytest` or `python -m pytest` in the project root.

### Q: How do I update the vector database with new Wikipedia data?
A: Run the data fetch and embedding scripts in the `scripts/` directory.

---

# ⚠️ Error Codes

| Code | Meaning                  | Typical Cause                        |
|------|--------------------------|--------------------------------------|
| 200  | Success                  | Request processed successfully        |
| 400  | Bad Request              | Invalid input, missing fields         |
| 404  | Not Found                | Endpoint does not exist               |
| 500  | Internal Server Error    | Server-side error, see logs           |

---

# 🧪 Testing

- **Run all tests:**
  ```bash
  python -m pytest -v
  ```
- **Run API integration tests:**
  ```bash
  python -m pytest -v tests/test_api_integration.py
  ```
- **Run coverage:**
  ```bash
  poetry run pytest --cov=app --cov=scripts --cov-report=xml --cov-report=html
  ```

---

# 🐳 Docker & Deployment

## Build and Run with Docker
```bash
docker build -t wiki-search-api .
docker run -p 8000:8000 --env-file .env wiki-search-api
```

## Using Docker Compose
```bash
docker compose up --build
```

## Deploying to Cloud (e.g., AWS ECS/App Runner)
- Build and push Docker image to ECR
- Deploy using ECS Fargate or App Runner (see README for details)

---

# ⚙️ Configuration

- All configuration is managed via `.env` (see `env.example`)
- Required: `COHERE_API_KEY`
- Other settings: vector DB path, chunk size, language, etc.

---

## Development Guidelines
- Use Poetry for dependency management
- Follow PEP 8 code style
- Write clear comments and documentation
- Add/maintain tests for all features

---

# 📈 Roadmap

- [x] Wikipedia data fetch & cleaning
- [x] Vectorization & FAISS storage
- [x] Core RAG workflow (LangGraph)
- [x] API endpoints (FastAPI)
- [x] Logging & session tracking
- [x] Guardrails validation
- [x] Unit & integration tests
- [x] Dockerization
- [x] API documentation improvements
- [ ] Monitoring & alerting
- [ ] Performance optimization
- [ ] User management & caching
- [ ] Cloud deployment (AWS/GCP/Azure)

---

# 📄 License

MIT License

---

# 📬 Contact

For questions or suggestions, please open an issue.