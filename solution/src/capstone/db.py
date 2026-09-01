"""Database configuration and connection helpers.

Centralises how every phase reaches Postgres so connection details live in
exactly one place (`.env`). Uses psycopg 3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    user: str
    password: str
    database: str
    schema: str
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("CAPSTONE_DATA_DIR", "../objectives/data"))
        if not data_dir.is_absolute():
            data_dir = (REPO_ROOT / data_dir).resolve()
        return cls(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", ""),
            database=os.environ.get("PGDATABASE", "capstone_hospital_analytics"),
            schema=os.environ.get("CAPSTONE_SCHEMA", "capstone_solution"),
            data_dir=data_dir,
        )

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.database}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


SETTINGS = Settings.from_env()


def connect(*, autocommit: bool = False, search_path: bool = True) -> psycopg.Connection:
    """Open a connection with the capstone schema on the search_path."""
    conn = psycopg.connect(SETTINGS.dsn, autocommit=autocommit)
    if search_path:
        with conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{SETTINGS.schema}", public')
        if not autocommit:
            conn.commit()
    return conn


def engine(*, search_path: bool = True) -> Engine:
    """SQLAlchemy engine (used by pandas.read_sql)."""
    connect_args = {}
    if search_path:
        connect_args["options"] = f"-csearch_path={SETTINGS.schema},public"
    return create_engine(SETTINGS.sqlalchemy_url, connect_args=connect_args)
