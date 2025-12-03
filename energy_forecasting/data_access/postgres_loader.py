from __future__ import annotations

import os

from sqlalchemy import create_engine
import pandas as pd

from energy_forecasting.entity.config_entity import DatabaseConfig
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger


def build_postgres_url(cfg: DatabaseConfig) -> str:
    """
    Build a SQLAlchemy connection URL for Postgres.

    Environment variables (if set) override cfg.
    - Local venv: it uses cfg.
    - Docker: we set ENERGY_DB_HOST=energy-db.
    """
    host = os.getenv("ENERGY_DB_HOST", cfg.host)
    port = os.getenv("ENERGY_DB_PORT", str(cfg.port))
    user = os.getenv("ENERGY_DB_USER", cfg.user)
    password = os.getenv("ENERGY_DB_PASSWORD", cfg.password)
    db_name = os.getenv("ENERGY_DB_NAME", cfg.db_name)

    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    logger.info("Using Postgres URL: %s", conn_str)
    return conn_str


def load_recent_table_from_db(db_config: DatabaseConfig) -> pd.DataFrame:
    conn_str = build_postgres_url(db_config)

    # For logging, resolve what we actually use
    host = os.getenv("ENERGY_DB_HOST", db_config.host)
    port = os.getenv("ENERGY_DB_PORT", str(db_config.port))
    db_name = os.getenv("ENERGY_DB_NAME", db_config.db_name)

    logger.info(
        "Connecting to Postgres %s:%s db=%s, table=%s (recent window: %d hours)",
        host,
        port,
        db_name,
        db_config.table,
        db_config.hours_history,
    )

    engine = create_engine(conn_str)
    sql = f"""
        SELECT *
        FROM {db_config.table}
        WHERE ds >= (
            SELECT MAX(ds) FROM {db_config.table}
        ) - INTERVAL '{db_config.hours_history} hour'
        ORDER BY ds
    """

    conn = None
    try:
        conn = engine.raw_connection()
        df = pd.read_sql(sql, conn)
    except Exception as e:
        raise EnergyException(e) from e
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    logger.info(
        "Loaded %d rows from DB (last %d hours).",
        len(df),
        db_config.hours_history,
    )

    return df


def load_full_table_from_db(db_config: DatabaseConfig) -> pd.DataFrame:
    conn_str = build_postgres_url(db_config)

    host = os.getenv("ENERGY_DB_HOST", db_config.host)
    port = os.getenv("ENERGY_DB_PORT", str(db_config.port))
    db_name = os.getenv("ENERGY_DB_NAME", db_config.db_name)

    logger.info(
        "Connecting to Postgres %s:%s db=%s, table=%s (full table load)",
        host,
        port,
        db_name,
        db_config.table,
    )

    engine = create_engine(conn_str)
    sql = f"SELECT * FROM {db_config.table}"

    conn = None
    try:
        conn = engine.raw_connection()
        df = pd.read_sql(sql, conn)
    except Exception as e:
        raise EnergyException(e) from e
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    logger.info("Loaded %d rows from DB (full table).", len(df))
    return df
