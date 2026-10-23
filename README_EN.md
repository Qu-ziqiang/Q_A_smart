<div align="center">
  <img src="assets/logo/logo.png" alt="SmartImager Logo" width="200">
  <h1>SmartImager</h1>

  <p>Intelligent Image Search / Management & AI Conversational Recommendation System</p>

  <div>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-orange.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/React-18-61DAFB.svg" alt="React">
  </div>

  <div>
    <a href="README.md">中文</a> |
    <a href="README_EN.md">English</a>
  </div>
</div>

## Project Overview

SmartImager is a modern intelligent image search and management system, built with a lightweight integrated FastAPI backend and React frontend. It features vector semantic retrieval, image-to-image search, fuzzy search, and conversational AI recommendations. Vector storage is **PostgreSQL + pgvector (HNSW / cosine)**; the encoders are **CLIP**: text `clip-ViT-B-32-multilingual-v1` (512-dim) and image `openai/clip-vit-base-patch32` (512-dim).

## 🏗️ System Architecture

### Architecture Overview

![System Architecture](docs/架构图.png)
The current version uses an integrated backend (FastAPI) + frontend (React) architecture. The AI recommendation/conversational agent is integrated in `backend/ai_func/recommendation`, combining PostgreSQL + pgvector vector retrieval and SSE streaming output. No extra AI microservice is required for intelligent search and multi-turn recommendations. The database can be started with `docker-compose.yml` (`pgvector/pgvector:pg16`).

## Features

### 🖼️ Image Management

- **Image Browsing** - Intuitive grid layout and image preview
- **Tag System** - Powerful multi-tag classification and management
- **Metadata Editing** - Flexible title, description, and custom tag management
- **Batch Operations** - Efficient batch upload, analysis, and tag management
- **Drag & Drop Upload** - Supports multi-file selection and drag-and-drop upload

### 🔍 Intelligent Search

- **Text Search** - Image retrieval based on natural language descriptions
- **Image Search** - Find similar content using image-to-image search
- **Vector Search** - Multi-dimensional search based on title, description, and image content
- **Similarity Search** - Retrieve similar images based on reference images
- **Filter Search** - Supports tag and time-based combination filtering
- **Fuzzy Search (Fuzzy LIKE)** - Lightweight keyword LIKE matching for fast search

#### 💬 Chat-driven Image Retrieval

Multi-turn conversational intelligent image retrieval and recommendation:

- **Context Memory**: Supports 64K rolling window conversation history, auto-trims to keep key info
- **Smart Query Rewriting**: Normalizes user input/keyword extraction to improve vector retrieval accuracy
- **Phased Process**: Rewrite -> Multi-vector retrieval (title/desc/image) -> Result re-ranking -> Generate reply
- **SSE Streaming Output**: Events include `rewrite_start` / `assistant_delta` / `complete` / `error`
- **Session Management**: Create/list/delete sessions, `conversation_id` binds context
- **Result Enhancement**: Returns brief image info (id/score/title/tags/public_url) + selected image ID list

Main APIs:

| Function | Method | Path |
|----------|--------|------|
| Create Session | POST | `/api/v1/ai/conversations/create` |
| List Sessions | GET  | `/api/v1/ai/conversations` |
| Delete Session | DELETE | `/api/v1/ai/conversations/{conversation_id}` |
| Get Session Messages | GET | `/api/v1/ai/conversations/{conversation_id}/messages` |
| Chat Recommendation (single) | POST | `/api/v1/ai/recommend/chat` |
| Chat Recommendation (SSE stream) | POST | `/api/v1/ai/recommend/chat/stream` |

Request Example (streaming; use the actual port printed on startup, or set via SIF_PORT or root start_config.yaml):

```bash
# Note: replace 8000 with your actual backend port (or set via SIF_PORT)
curl -N -X POST http://localhost:8000/api/v1/ai/recommend/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "demo-session-1",
    "query": "Find me some photos of city buildings under blue sky",
    "vector_targets": ["title_vector", "desc_vector", "image_vector"],
    "limit": 12
  }'
```

SSE Key Events (example):

```text
event: rewrite_start
data: {"message":"start","conversation_id":"demo-session-1"}

event: assistant_delta
data: {"delta":"Searching for relevant images..."}

event: complete
data: {"image_ids":[12,8,5,...],"assistant_text":"Found...","images_brief":[...]} 
```

Frontend can render "AI thinking / incremental reply / show image results" in real time based on event type for smooth interaction.

### 🤖 AI Analysis Features

- **Auto Analysis** - Image content understanding based on CLIP model
- **Smart Annotation** - Auto-generate image title, description, and tags
- **Batch Processing** - Supports large-scale batch AI analysis
- **API Integration** - Supports various multimodal vision model APIs

### 🎨 User Interface

- **Modern Design** - Beautiful UI based on React 18 + Ant Design 5
- **Responsive Layout** - Perfect for desktop and mobile
- **Real-time Interaction** - Supports image preview, zoom, and editing
- **Status Monitoring** - Real-time progress and system status
- **System Management** - Complete config and monitoring interface

## 🎯 Core Technical Features

- **🧠 Conversational Recommendation Agent** - Multi-turn context (64K rolling window) + tool function calls
- **🛡️ Vector Target Whitelist** - Prevents illegal table names/injection errors
- **🔌 Streaming SSE Output** - AI recommendation process pushes rewrite/search/result for better UX
- **⚡ High-performance Vector Retrieval** - PostgreSQL + pgvector (HNSW / cosine); metadata filtering and vector search run in the same database and the same SQL. Measured ~65 ms mean retrieval on a 1000-image gallery (`r1_eval_pg1000.json`)
- **🧠 Advanced AI Models** - CLIP multimodal encoders (text `clip-ViT-B-32-multilingual-v1` / image `clip-vit-base-patch32`, both 512-dim)
- **🎨 Modern Tech Stack** - React 18 + TypeScript + FastAPI for code quality and dev experience
- **🔧 Smart Startup Management** - One-click startup script, auto environment and dependency setup
- **🔄 Vector Migration & Rebuild** - `scripts/migrate_sqlite_to_pg.py` migrates an existing SQLite (sqlite-vec) database into PostgreSQL; `scripts/rebuild_pg_vectors.py` rebuilds all three vector tables with the current embedding model.

### 🔄 Embedding Model Migration

The data layer has been migrated to **PostgreSQL + pgvector**. Migration / rebuild flow:

```bash
docker compose up -d                                                     # 1. start pgvector (pg16)
python scripts/migrate_sqlite_to_pg.py --sqlite data/db/smartimager.db   # 2. migrate legacy SQLite DB (optional)
python scripts/rebuild_pg_vectors.py                                     # 3. rebuild vectors with current MODEL_PATH
python scripts/smoke_pg_search.py                                        # 4. smoke-test retrieval
```

Key facts:

- Schema: three `vector(512)` tables + `HNSW (vector_cosine_ops)` indexes, created idempotently by `init_db()`
- Write path: `INSERT ... ON CONFLICT (image_id) DO UPDATE`, vectors bound as `%s::vector` literals
- Retrieval: SQL `<=>` (cosine distance, score = 1 - distance); metadata filtering done in-DB via JOIN
- Migration/rebuild history and progress are recorded in the `model_migrations` table

> The legacy SQLite-era `migrate_embeddings.py` and `docs/model_migration.md` are only relevant when operating the old SQLite database.

## 🚀 Quick Start


git clone https://github.com/li-xiu-qi/SmartImageFinder.git

### Environment Initialization (Recommended: Manual Dependency Installation, Model Download via Script)

> Installing dependencies and downloading models may take a long time. Manual step-by-step installation is recommended.

```bash
# Clone project
git clone https://github.com/li-xiu-qi/SmartImageFinder.git
cd SmartImageFinder

# Manually install Python dependencies
pip install -r requirements.txt

# Start PostgreSQL + pgvector (must be available before the backend starts)
docker compose up -d

# Install Node.js dependencies (frontend)
cd frontend
npm install
cd ..
```

> ⚠️ Model download is recommended via `python start.py init`. Otherwise, manually download and place the model files in the `models/` directory. Model download is slow, please prepare in advance.

For details on model download, see docs/model_migration.md or relevant README sections.

### Start Service

#### Method 1: Use start.py for one-click startup (auto install dependencies, download models, generate config)
```bash
python start.py init
python start.py
```

#### Method 2: Manually start backend (recommended)
```bash
python main.py
```

#### Method 3: Manually start frontend
```bash
cd frontend
npm run dev
```

> Manual installation of dependencies and models is recommended to avoid long waiting times during automatic script installation.

### Ports and Proxy Configuration (Important)

- Backend startup precedence: CLI > environment variables > defaults.
  - CLI supported: `python main.py --host 0.0.0.0 --port 8000 --reload`
  - Environment variables:
    - `SIF_HOST` (default `0.0.0.0`)
    - `SIF_PORT` (default `8000`)
    - `SIF_RELOAD` (`1/true` enables hot reload)
- Frontend dev server port:
  - `SIF_FRONTEND_PORT` (default `5173`), read by `frontend/vite.config.ts`
  - Frontend->backend proxy origin: `SIF_BACKEND_ORIGIN` (falls back to `http://{SIF_HOST}:{SIF_PORT}`)
- The one-click starter `start.py` reads optional root `start_config.yaml` and sets the env vars above. Example:

```yaml
# start_config.yaml example
backend_host: 0.0.0.0
backend_port: 10060
reload: true
frontend_port: 5176
# Optional: explicitly set the frontend proxy backend origin
backend_origin: http://localhost:10060
```



## Application Scenarios

- **Personal Image Management** - Organize and retrieve personal photo library
- **Design Asset Management** - Efficiently manage and search design resources
- **Content Creation** - Provide intelligent image search for creators

## 🛠️ Technical Architecture

### Main Service Tech Stack

#### Backend (`backend/`)

- **FastAPI** - High-performance async web framework
- **PostgreSQL + pgvector** - Vector database (HNSW / cosine, three `vector(512)` tables); started via `docker-compose.yml`
- **CLIP** - Multimodal vector encoders (text `clip-ViT-B-32-multilingual-v1` / image `clip-vit-base-patch32`, 512-dim)
- **Connection Pool Management** - Thread-safe psycopg2 connection pool
- **Vector Cache** - diskcache-based vector caching system
- **Smart Re-ranking** - Intelligent image recommendation
- **Multimodal Analysis** - Image content understanding and analysis

#### Frontend (`frontend/`)

- **React 18 + TypeScript** - Modern frontend framework
- **Ant Design 5.25** - Enterprise UI component library
- **Vite** - Fast build tool
- **React Router 7.5** - Routing management
- **Axios** - HTTP client

### Core Components

- **Vector Search Engine** - Unified semantic search (text/image/direct vector/similarity)
- **AI Analysis Engine** - Auto image content analysis and annotation
- **Tag Management System** - Smart tag classification and management
- **Cache System** - High-performance vector cache
- **Config Management** - Flexible system config management

## 🖥️ System Requirements

### Supported Platforms

- **Windows**: x86_64
- **Linux**: x86_64, aarch64  
- **macOS**: x86_64 (Intel), aarch64 (Apple Silicon)

## 📁 Project Structure

```
SmartImager/
├── backend/                       # Backend main service (API, DB, AI recommendation, etc.)
│   ├── routers/                   # Routers (images/tags/search/metadata, etc.)
│   ├── db_func/                   # DB and vector operations
│   ├── ai_func/                   # AI analysis & recommendation (incl. recommendation/agent.py)
│   ├── config/                    # Config files & drivers
│   └── global_schemas.py          # Common response models
├── frontend/                      # Frontend main service (React+AntD)
│   ├── src/pages/                 # Page components
│   ├── src/components/            # Common components
│   ├── src/services/              # API service layer
│   └── public/                    # Static assets
├── models/                        # AI model files (local or downloaded)
├── data/                          # Data storage
│   ├── db/                        # Legacy SQLite DB files (migrated to PostgreSQL; kept for backup/migration only)
│   ├── images/                    # Image storage
│   ├── caches/                    # Vector cache
│   └── temp/                      # Temp files
├── scripts/                       # Startup & environment management scripts
├── docs/                          # Architecture & API docs
├── requirements.txt               # Python dependencies
├── main.py                        # Backend main entry (API service)
├── start.py                       # One-click startup entry
└── README.md                      # Project description
```

## 🔧 Detailed Configuration

### Service Ports

- **Backend**: 8000 (`main.py` default; override via `SIF_PORT` env var or CLI `--port`)
- **Frontend**: 5173 (Vite default)

### Config File

Main config file: `backend/config/files/config.yaml`:

```yaml
MODEL_PATH: sentence-transformers/clip-ViT-B-32-multilingual-v1  # text CLIP model (512-dim)
EMBEDDING_DIMENSION: 512         # must match the vector(512) columns in PostgreSQL
DATABASE_URL: postgresql://smartimager:smartimager@127.0.0.1:5432/smartimager
PG_HOST: 127.0.0.1
PG_PORT: 5432
PG_USER: smartimager
PG_PASSWORD: smartimager
PG_DATABASE: smartimager
UPLOAD_DIR: ./data/images        # image upload dir
TEXT_VECTOR_CACHE_DIR: ./data/caches/text_vector_cache
IMAGE_VECTOR_CACHE_DIR: ./data/caches/image_vector_cache
CHAT_MODEL: deepseek-chat        # chat / planning model
VISION_MODEL: "null"             # vision model off by default
```

### 💡 Usage Tips

#### Service Access

- **Frontend UI**: <http://localhost:5173>
- **Backend API**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>

## 📊 System Monitoring & Management

### System Status Monitoring

Via system settings page, you can monitor:

- **System Info** - CPU, memory, disk usage
- **DB Status** - Connection pool, table stats
- **Storage Info** - Image count, tag count, storage usage
- **Cache Status** - Cache dir size (MB), optional cache.db size, last_scan
- **Vector DB** - pgvector extension version, HNSW index status, vector table row counts

### Cache Statistics

Cache API `/api/v1/system/cache` now only returns dir size (MB), optional cache.db size, and last_scan; after cleaning, frontend polls `/api/v1/system/cache/brief` to confirm reset.

### Conversational Recommendation SSE Events

Streaming endpoint only sends events: `rewrite_start`, `assistant_delta` (multiple), `complete`, `error`; internal selection is aggregated in complete.

## License

This project is licensed under [Apache License 2.0](LICENSE).

## Contact

If you have any questions or suggestions, feel free to contact me:

<div align="center">
  <img src="assets/wechat/筱可AI研习社_258.jpg" alt="XiaoKe AI Study Club" width="200">
  <p>Scan the QR code to follow "XiaoKe AI Study Club" WeChat Official Account</p>
</div>
