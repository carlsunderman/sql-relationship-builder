"""Tests for admin/config_store.py -- SQLite config store with encryption and auth."""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

# Import the module under test
from admin.config_store import (
    PROVIDER_AZURE,
    PROVIDER_LOCAL,
    PROVIDER_ANTHROPIC,
    ALL_PROVIDERS,
    ServerConnection,
    ProviderConfig,
    encrypt_value,
    decrypt_value,
    hash_password,
    check_password,
    init_db,
    list_configs,
    get_config,
    save_config,
    delete_config,
    get_active_config,
    set_admin_password,
    verify_admin_password,
    has_admin_password,
    to_llm_config,
    list_server_connections,
    get_server_connection,
    save_server_connection,
    delete_server_connection,
    get_connection_password,
    get_allowed_databases,
)


@pytest.fixture()
def db_path(tmp_path):
    """Provide a temporary DB path and patch ADMIN_DB_PATH."""
    path = str(tmp_path / "test_admin.db")
    with patch("admin.config_store.ADMIN_DB_PATH", path):
        yield path


@pytest.fixture()
def encryption_key(tmp_path):
    """Provide a temporary encryption key path and patch ENCRYPTION_KEY_PATH."""
    from cryptography.fernet import Fernet
    key_path = str(tmp_path / ".encryption_key")
    key = Fernet.generate_key().decode()
    with patch("admin.config_store.ENCRYPTION_KEY_PATH", key_path), \
         patch.dict(os.environ, {"ADMIN_ENCRYPTION_KEY": key}):
        yield key


@pytest.fixture()
def setup_db(db_path, encryption_key):
    """Initialize a fresh DB for each test."""
    init_db()


# ---------------------------------------------------------------------------
# Encryption / Decryption
# ---------------------------------------------------------------------------

class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self, encryption_key):
        original = "sk-test-api-key-12345"
        encrypted = encrypt_value(original)
        assert encrypted != original
        decrypted = decrypt_value(encrypted)
        assert decrypted == original

    def test_decrypt_invalid_token(self, encryption_key, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            result = decrypt_value("not-valid-fernet-token")
            assert result == ""
            assert "Failed to decrypt" in caplog.text


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

class TestPassword:
    def test_hash_and_check(self):
        password = "secure-password-123"
        hashed = hash_password(password)
        assert hashed != password
        assert check_password(password, hashed) is True
        assert check_password("wrong-password", hashed) is False

    def test_set_and_verify_admin_password(self, setup_db):
        assert has_admin_password() is False
        set_admin_password("admin-pass")
        assert has_admin_password() is True
        assert verify_admin_password("admin-pass") is True
        assert verify_admin_password("wrong") is False


# ---------------------------------------------------------------------------
# ProviderConfig CRUD
# ---------------------------------------------------------------------------

class TestProviderConfigCRUD:
    def test_create_and_list(self, setup_db):
        config = ProviderConfig(name="Test Azure", provider=PROVIDER_AZURE, is_active=True)
        save_config(config)
        configs = list_configs()
        assert len(configs) == 1
        assert configs[0].name == "Test Azure"

    def test_get_by_id(self, setup_db):
        config = ProviderConfig(name="Local Test", provider=PROVIDER_LOCAL)
        save_config(config)
        retrieved = get_config(config.id)
        assert retrieved is not None
        assert retrieved.name == "Local Test"

    def test_get_nonexistent(self, setup_db):
        assert get_config("nonexistent-id") is None

    def test_update_existing(self, setup_db):
        config = ProviderConfig(name="Original", provider=PROVIDER_LOCAL)
        save_config(config)
        config.name = "Updated"
        config.model = "claude-sonnet-4"
        save_config(config)
        retrieved = get_config(config.id)
        assert retrieved.name == "Updated"
        assert retrieved.model == "claude-sonnet-4"

    def test_delete(self, setup_db):
        config = ProviderConfig(name="To Delete", provider=PROVIDER_LOCAL)
        save_config(config)
        delete_config(config.id)
        assert get_config(config.id) is None
        assert len(list_configs()) == 0

    def test_only_one_active(self, setup_db):
        c1 = ProviderConfig(name="First", provider=PROVIDER_LOCAL, is_active=True)
        save_config(c1)
        c2 = ProviderConfig(name="Second", provider=PROVIDER_AZURE, is_active=True)
        save_config(c2)
        active = get_active_config()
        assert active is not None
        assert active.id == c2.id
        # First should no longer be active
        c1_reloaded = get_config(c1.id)
        assert not c1_reloaded.is_active

    def test_get_active_returns_none_when_empty(self, setup_db):
        assert get_active_config() is None

    def test_all_provider_types(self, setup_db):
        for provider in ALL_PROVIDERS:
            config = ProviderConfig(name=f"Test {provider}", provider=provider)
            save_config(config)
        assert len(list_configs()) == len(ALL_PROVIDERS)


# ---------------------------------------------------------------------------
# ProviderConfig Serialization
# ---------------------------------------------------------------------------

class TestProviderConfigSerialization:
    def test_to_dict_masks_api_key(self, setup_db, encryption_key):
        config = ProviderConfig(
            name="Test",
            provider=PROVIDER_AZURE,
            api_key_encrypted=encrypt_value("secret-key"),
        )
        d = config.to_dict(mask_api_key=True)
        assert d["api_key"] == "****"
        assert "api_key_encrypted" not in d

    def test_to_dict_unmasks_api_key(self, setup_db, encryption_key):
        config = ProviderConfig(
            name="Test",
            provider=PROVIDER_AZURE,
            api_key_encrypted=encrypt_value("secret-key"),
        )
        d = config.to_dict(mask_api_key=False)
        assert d["api_key"] == "secret-key"

    def test_to_dict_parsing_extra_config(self, setup_db):
        config = ProviderConfig(
            name="Test",
            provider=PROVIDER_ANTHROPIC,
            extra_config=json.dumps({"anthropic_version": "2023-06-01"}),
        )
        d = config.to_dict()
        assert d["extra_config_parsed"] == {"anthropic_version": "2023-06-01"}
        assert "extra_config" not in d

    def test_from_dict_roundtrip(self, setup_db, encryption_key):
        original = ProviderConfig(
            name="Roundtrip",
            provider=PROVIDER_LOCAL,
            endpoint="http://localhost:8080",
            model="llama3",
            temperature=0.5,
            max_tokens=8192,
            extra_config=json.dumps({"custom": True}),
        )
        save_config(original)
        d = original.to_dict(mask_api_key=False)
        restored = ProviderConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.provider == original.provider
        assert restored.endpoint == original.endpoint
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens


# ---------------------------------------------------------------------------
# to_llm_config Bridge
# ---------------------------------------------------------------------------

class TestToLlmConfig:
    def test_bridge_produces_llm_config_dict(self, setup_db):
        pc = ProviderConfig(
            name="Bridge Test",
            provider=PROVIDER_LOCAL,
            endpoint="http://localhost:11434",
            model="llama3",
            temperature=0.3,
            max_tokens=4096,
            timeout=60,
        )
        save_config(pc)
        d = to_llm_config(pc)
        assert d["provider"] == PROVIDER_LOCAL
        assert d["endpoint"] == "http://localhost:11434"
        assert d["model"] == "llama3"
        assert d["temperature"] == 0.3
        assert d["max_tokens"] == 4096
        assert d["timeout"] == 60


# ---------------------------------------------------------------------------
# Server Connection CRUD
# ---------------------------------------------------------------------------

class TestServerConnections:
    def test_create_and_list(self, setup_db):
        sc = ServerConnection(
            name="Prod SQL",
            server="sql.example.com",
            username="reader",
        )
        save_server_connection(sc)
        connections = list_server_connections()
        assert len(connections) == 1
        assert connections[0].name == "Prod SQL"

    def test_get_by_id(self, setup_db):
        sc = ServerConnection(name="Test", server="localhost")
        save_server_connection(sc)
        retrieved = get_server_connection(sc.id)
        assert retrieved is not None
        assert retrieved.server == "localhost"

    def test_get_nonexistent(self, setup_db):
        assert get_server_connection("no-such-id") is None

    def test_delete(self, setup_db):
        sc = ServerConnection(name="ToDelete", server="localhost")
        save_server_connection(sc)
        delete_server_connection(sc.id)
        assert get_server_connection(sc.id) is None

    def test_password_encryption_roundtrip(self, setup_db, encryption_key):
        sc = ServerConnection(
            name="Encrypted",
            server="sql.example.com",
            username="admin",
            password_encrypted=encrypt_value("db-pass-123"),
        )
        save_server_connection(sc)
        retrieved = get_server_connection(sc.id)
        password = get_connection_password(retrieved)
        assert password == "db-pass-123"

    def test_allowed_databases(self, setup_db):
        sc = ServerConnection(
            name="Filtered",
            server="sql.example.com",
            allowed_databases=json.dumps(["db1", "db2"]),
        )
        save_server_connection(sc)
        retrieved = get_server_connection(sc.id)
        dbs = get_allowed_databases(retrieved)
        assert dbs == ["db1", "db2"]

    def test_to_dict_masks_password(self, setup_db, encryption_key):
        sc = ServerConnection(
            name="Masked",
            server="sql.example.com",
            password_encrypted=encrypt_value("secret"),
        )
        d = sc.to_dict(mask_password=True)
        assert d["password"] == "****"
        assert "password_encrypted" not in d
