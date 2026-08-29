"""SQLite-backed config store with encrypted API keys and password auth.

Stores AI provider configurations. API keys are encrypted with Fernet.
Admin password is hashed with bcrypt.
"""

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
import bcrypt

logger = logging.getLogger(__name__)

PROVIDER_AZURE = "azure"
PROVIDER_LOCAL = "local"
PROVIDER_ANTHROPIC = "anthropic"
ALL_PROVIDERS = [PROVIDER_AZURE, PROVIDER_LOCAL, PROVIDER_ANTHROPIC]

ADMIN_DB_PATH = os.path.join(os.path.dirname(__file__), "admin_config.db")
ENCRYPTION_KEY_PATH = os.path.join(os.path.dirname(__file__), ".encryption_key")

# delete me!!
import dotenv
dotenv.load_dotenv(
   "../env"
)
MONGODB_URI = os.getenv("MONGODB_URI")

# end of delete

# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def _get_fernet() -> Fernet:
    """Return a Fernet cipher, creating the key file if needed."""
    key_env = os.environ.get("ADMIN_ENCRYPTION_KEY")
    if key_env:
        # Validate it's a valid base64 Fernet key.
        try:
            key_env.encode().strip()
            Fernet(key_env.encode())
        except Exception:
            raise ValueError(
                "ADMIN_ENCRYPTION_KEY is set but is not a valid Fernet key. "
                "Generate one with: python -c "
                '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
            )
        return Fernet(key_env.encode())

    if os.path.exists(ENCRYPTION_KEY_PATH):
        with open(ENCRYPTION_KEY_PATH, "r") as f:
            return Fernet(f.read().strip().encode())

    # Auto-generate.
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_PATH, "w") as f:
        f.write(key.decode())
    logger.info("Generated new encryption key at %s", ENCRYPTION_KEY_PATH)
    return Fernet(key)


def encrypt_value(value: str) -> str:
    """Encrypt a plaintext value. Returns empty string for empty input."""
    if not value:
        return ""
    return _get_fernet().encrypt(value.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt an encrypted value. Returns empty string for empty input."""
    if not encrypted:
        return ""
    try:
        return _get_fernet().decrypt(encrypted.encode()).decode()
    except InvalidToken:
        logger.warning("Failed to decrypt value; encryption key may have changed")
        return ""


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerConnection:
    """One preconfigured SQL Server connection."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # human-readable label, e.g. "Production SQL"
    server: str = ""  # hostname or IP
    username: str = ""  # SQL Server login
    password_encrypted: str = ""  # Fernet-encrypted password
    allowed_databases: str = "[]"  # JSON list of database names; [] = discover all

    def to_dict(self, mask_password: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if mask_password:
            d["password"] = "****" if self.password_encrypted else ""
        else:
            d["password"] = decrypt_value(self.password_encrypted)
        d["allowed_databases_parsed"] = json.loads(self.allowed_databases or "[]")
        del d["password_encrypted"]
        del d["allowed_databases"]
        return d


SERVER_CONNECTION_COLUMNS = [
    "id", "name", "server", "username", "password_encrypted", "allowed_databases",
]


@dataclass
class ProviderConfig:
    """One named AI provider configuration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # human-readable label, e.g. "Production Azure"
    provider: str = PROVIDER_LOCAL
    endpoint: str = ""
    model: str = ""
    api_key_encrypted: str = ""
    api_version: str = "2024-02-01"
    deployment_name: str = ""
    temperature: float = 0.0
    max_tokens: int = 16000
    timeout: int = 120
    verify_ssl: bool = True
    extra_config: str = "{}"  # JSON blob for provider-specific fields
    is_active: bool = False

    def to_dict(self, mask_api_key: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if mask_api_key:
            d["api_key"] = "****" if self.api_key_encrypted else ""
        else:
            d["api_key"] = decrypt_value(self.api_key_encrypted)
        d["extra_config_parsed"] = json.loads(self.extra_config or "{}")
        del d["api_key_encrypted"]
        del d["extra_config"]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        api_key = data.pop("api_key", "")
        extra = data.pop("extra_config_parsed", {})
        return cls(
            api_key_encrypted=encrypt_value(api_key) if api_key else "",
            extra_config=json.dumps(extra),
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS provider_configs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key_encrypted TEXT NOT NULL DEFAULT '',
    api_version TEXT NOT NULL DEFAULT '2024-02-01',
    deployment_name TEXT NOT NULL DEFAULT '',
    temperature REAL NOT NULL DEFAULT 0.0,
    max_tokens INTEGER NOT NULL DEFAULT 16000,
    timeout INTEGER NOT NULL DEFAULT 120,
    verify_ssl INTEGER NOT NULL DEFAULT 1,
    extra_config TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS admin_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    server TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    password_encrypted TEXT NOT NULL DEFAULT '',
    allowed_databases TEXT NOT NULL DEFAULT '[]'
);
"""

COLUMNS = [
    "id", "name", "provider", "endpoint", "model", "api_key_encrypted",
    "api_version", "deployment_name", "temperature", "max_tokens",
    "timeout", "verify_ssl", "extra_config", "is_active",
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def _row_to_config(row: sqlite3.Row) -> ProviderConfig:
    return ProviderConfig(
        **{col: row[col] for col in COLUMNS}
    )


def list_configs() -> List[ProviderConfig]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM provider_configs ORDER BY is_active DESC, name"
    ).fetchall()
    conn.close()
    return [_row_to_config(r) for r in rows]


def get_config(config_id: str) -> Optional[ProviderConfig]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM provider_configs WHERE id = ?", (config_id,)
    ).fetchone()
    conn.close()
    return _row_to_config(row) if row else None


def save_config(config: ProviderConfig) -> ProviderConfig:
    """Insert or update a config. If set active, deactivate others."""
    conn = _get_conn()
    if config.is_active:
        conn.execute("UPDATE provider_configs SET is_active = 0")

    conn.execute(
        """INSERT INTO provider_configs
           (id, name, provider, endpoint, model, api_key_encrypted,
            api_version, deployment_name, temperature, max_tokens,
            timeout, verify_ssl, extra_config, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name, provider=excluded.provider,
             endpoint=excluded.endpoint, model=excluded.model,
             api_key_encrypted=excluded.api_key_encrypted,
             api_version=excluded.api_version,
             deployment_name=excluded.deployment_name,
             temperature=excluded.temperature,
             max_tokens=excluded.max_tokens,
             timeout=excluded.timeout,
             verify_ssl=excluded.verify_ssl,
             extra_config=excluded.extra_config,
             is_active=excluded.is_active
        """,
        (
            config.id, config.name, config.provider, config.endpoint,
            config.model, config.api_key_encrypted, config.api_version,
            config.deployment_name, config.temperature, config.max_tokens,
            config.timeout, int(config.verify_ssl), config.extra_config,
            int(config.is_active),
        ),
    )
    conn.commit()
    conn.close()
    return config


def delete_config(config_id: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM provider_configs WHERE id = ?", (config_id,))
    conn.commit()
    conn.close()


def get_active_config() -> Optional[ProviderConfig]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM provider_configs WHERE is_active = 1 LIMIT 1"
    ).fetchone()
    conn.close()
    return _row_to_config(row) if row else None


# ---------------------------------------------------------------------------
# Admin password
# ---------------------------------------------------------------------------

def get_admin_password_hash() -> Optional[str]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT value FROM admin_settings WHERE key = 'admin_password'"
    ).fetchone()
    conn.close()
    return row["value"] if row else None


def set_admin_password(password: str) -> None:
    hashed = hash_password(password)
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO admin_settings (key, value) VALUES (?, ?)",
        ("admin_password", hashed),
    )
    conn.commit()
    conn.close()


def verify_admin_password(password: str) -> bool:
    hashed = get_admin_password_hash()
    if not hashed:
        return False
    return check_password(password, hashed)


def has_admin_password() -> bool:
    return get_admin_password_hash() is not None


# ---------------------------------------------------------------------------
# Env var seeding
# ---------------------------------------------------------------------------

def seed_from_env() -> Optional[ProviderConfig]:
    """If no configs exist and env vars are set, create a default config."""
    if list_configs():
        return None

    provider = os.environ.get("LLM_PROVIDER", PROVIDER_LOCAL).strip().lower()
    if provider not in ALL_PROVIDERS:
        provider = PROVIDER_LOCAL

    if provider == PROVIDER_AZURE:
        return _seed_azure()
    elif provider == PROVIDER_ANTHROPIC:
        return _seed_anthropic()
    else:
        return _seed_local()


def _seed_azure() -> Optional[ProviderConfig]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        return None

    return save_config(ProviderConfig(
        name="Azure OpenAI (from env)",
        provider=PROVIDER_AZURE,
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        model=os.environ.get("AZURE_OPENAI_MODEL", ""),
        api_key_encrypted=encrypt_value(api_key),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        verify_ssl=os.environ.get("LLM_VERIFY_SSL", "false").lower() not in ("false", "0", "no"),
        is_active=True,
    ))


def _seed_local() -> Optional[ProviderConfig]:
    endpoint = os.environ.get("LOCAL_OPENAI_ENDPOINT", "http://localhost:8080")
    api_key = os.environ.get("LOCAL_OPENAI_API_KEY", "")

    return save_config(ProviderConfig(
        name="Local (from env)",
        provider=PROVIDER_LOCAL,
        endpoint=endpoint,
        model=os.environ.get("LOCAL_OPENAI_MODEL", "llama3"),
        api_key_encrypted=encrypt_value(api_key) if api_key else "",
        api_version="2024-02-01",
        verify_ssl=os.environ.get("LLM_VERIFY_SSL", "false").lower() not in ("false", "0", "no"),
        is_active=True,
    ))


def _seed_anthropic() -> Optional[ProviderConfig]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    return save_config(ProviderConfig(
        name="Anthropic (from env)",
        provider=PROVIDER_ANTHROPIC,
        endpoint=os.environ.get("ANTHROPIC_ENDPOINT", "https://api.anthropic.com"),
        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        api_key_encrypted=encrypt_value(api_key),
        extra_config=json.dumps({
            "anthropic_version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        }),
        is_active=True,
    ))


# ---------------------------------------------------------------------------
# LLMConfig bridge (for app.py compatibility)
# ---------------------------------------------------------------------------

def to_llm_config(pc: ProviderConfig) -> Dict[str, Any]:
    """Convert a ProviderConfig to an LLMConfig-compatible dict."""
    extra = json.loads(pc.extra_config or "{}")
    return {
        "provider": pc.provider,
        "endpoint": pc.endpoint,
        "model": pc.model,
        "api_key": decrypt_value(pc.api_key_encrypted),
        "api_version": pc.api_version,
        "deployment_name": pc.deployment_name,
        "temperature": pc.temperature,
        "max_tokens": pc.max_tokens,
        "timeout": pc.timeout,
        "verify_ssl": pc.verify_ssl,
        "extra_config": extra,
    }


# ---------------------------------------------------------------------------
# Server Connections CRUD
# ---------------------------------------------------------------------------

def _row_to_server_connection(row: sqlite3.Row) -> ServerConnection:
    return ServerConnection(
        **{col: row[col] for col in SERVER_CONNECTION_COLUMNS}
    )


def list_server_connections() -> List[ServerConnection]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM server_connections ORDER BY name"
    ).fetchall()
    conn.close()
    return [_row_to_server_connection(r) for r in rows]


def get_server_connection(conn_id: str) -> Optional[ServerConnection]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM server_connections WHERE id = ?", (conn_id,)
    ).fetchone()
    conn.close()
    return _row_to_server_connection(row) if row else None


def save_server_connection(sc: ServerConnection) -> ServerConnection:
    """Insert or update a server connection."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO server_connections
           (id, name, server, username, password_encrypted, allowed_databases)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name,
             server=excluded.server,
             username=excluded.username,
             password_encrypted=excluded.password_encrypted,
             allowed_databases=excluded.allowed_databases
        """,
        (
            sc.id, sc.name, sc.server, sc.username,
            sc.password_encrypted, sc.allowed_databases,
        ),
    )
    conn.commit()
    conn.close()
    return sc


def delete_server_connection(conn_id: str) -> None:
    conn = _get_conn()
    conn.execute(
        "DELETE FROM server_connections WHERE id = ?", (conn_id,)
    )
    conn.commit()
    conn.close()


def get_connection_password(sc: ServerConnection) -> str:
    """Decrypt and return the stored password for a server connection."""
    return decrypt_value(sc.password_encrypted)


def get_allowed_databases(sc: ServerConnection) -> List[str]:
    """Return the list of allowed databases for a server connection."""
    return json.loads(sc.allowed_databases or "[]")
