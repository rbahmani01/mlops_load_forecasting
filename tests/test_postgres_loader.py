import pytest

from energy_forecasting.data_access.postgres_loader import build_postgres_url
from energy_forecasting.entity.config_entity import DatabaseConfig


def test_build_postgres_url_uses_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENERGY_DB_* env vars should override DatabaseConfig when building the URL."""
    cfg = DatabaseConfig(
        host="cfg-host",
        port=5432,
        db_name="cfg_db",
        user="cfg_user",
        password="cfg_pwd",
        table="df_mlf",
        hours_history=24,
    )

    # Only override host; keep others coming from cfg
    monkeypatch.setenv("ENERGY_DB_HOST", "env-host")
    monkeypatch.delenv("ENERGY_DB_PORT", raising=False)
    monkeypatch.delenv("ENERGY_DB_USER", raising=False)
    monkeypatch.delenv("ENERGY_DB_PASSWORD", raising=False)
    monkeypatch.delenv("ENERGY_DB_NAME", raising=False)

    url = build_postgres_url(cfg)

    assert "env-host" in url
    assert "cfg_db" in url
    assert "cfg_user" in url
    assert "cfg_pwd" in url
    assert url.startswith("postgresql+psycopg2://")
