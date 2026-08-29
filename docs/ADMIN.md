# AI Config Admin

Streamlit-based admin page for managing AI provider settings, saved SQL Server connections, and saved relationship graphs. Backed by SQLite storage with encrypted credentials.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Streamlit (admin/admin.py)                 │
│  └─ Login page (bcrypt password hash)       │
│  └─ Configs (create, edit, delete, activate)│
│  └─ Server Connections (saved SQL servers)  │
│  └─ Saved Graphs (list, edit, fork, delete) │
│  └─ Settings (change password)              │
├─────────────────────────────────────────────┤
│  ConfigStore (admin/config_store.py)        │
│  └─ SQLite (admin/admin_config.db)          │
│  └─ Fernet encryption for API keys          │
│  └─ bcrypt for admin password               │
├─────────────────────────────────────────────┤
│  Streamlit app (app.py)                     │
│  └─ Reads active config from DB on startup  │
│  └─ Falls back to env vars if DB empty      │
└─────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the admin panel

```bash
streamlit run admin/admin.py --server.port 8550
```

Open http://localhost:8550 in your browser.

On first run, you'll be prompted to set an admin password.

### 3. Start the main app (separate terminal)

```bash
streamlit run app.py
```

The main app reads the active AI config from the SQLite DB on startup.

## Files

| File | Purpose |
|------|---------|
| `admin/admin.py` | Streamlit admin UI (login, configs, connections, saved graphs, settings) |
| `admin/config_store.py` | SQLite storage with Fernet encryption + bcrypt auth |
| `admin/admin_config.db` | SQLite database (auto-created, gitignored) |
| `admin/.encryption_key` | Fernet encryption key (auto-generated, gitignored) |

## Tabs

| Tab | Purpose |
|-----|---------|
| Configs | List, create, delete AI provider configs; mark one active |
| Edit Config | Edit endpoint, model, API key, and generation parameters |
| Server Connections | Save SQL Server connections: server, user, password (Fernet-encrypted), allowed databases. The main app lists these in its connection picker |
| Saved Graphs | Manage graphs saved from the main app: view node/edge preview, edit metadata, export JSON, fork, delete (two-click confirm) |
| Settings | Change the admin password |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ADMIN_ENCRYPTION_KEY` | Fernet key (auto-generated to `.encryption_key` if not set) |

### Seeding from env vars

If no configs exist in the DB, the app will seed a default config from these env vars:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `azure`, `local`, or `anthropic` |
| `AZURE_OPENAI_ENDPOINT` | Azure resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure deployment name |
| `AZURE_OPENAI_API_VERSION` | Azure API version |
| `LOCAL_OPENAI_ENDPOINT` | Local server URL |
| `LOCAL_OPENAI_API_KEY` | Local server API key (optional) |
| `LOCAL_OPENAI_MODEL` | Local model name |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model name |

## Docker

```bash
# Start both services
docker compose up -d

# Main app: http://localhost:8501
# Admin: http://localhost:8550
```

## Supported Providers

| Provider | Description |
|----------|-------------|
| `azure` | Azure OpenAI |
| `local` | OpenAI-compatible (Ollama, LM Studio, vLLM) |
| `anthropic` | Anthropic Claude |
