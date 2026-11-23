import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import URL, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "conversations.sqlite3"


@contextmanager
def database_sessions() -> Iterator[sessionmaker[Session]]:
    path = os.getenv("AI_ASSISTANT_DB_PATH") or DEFAULT_DB_PATH
    engine = create_engine(URL.create("sqlite", database=str(path)))

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON").close()

    try:
        Base.metadata.create_all(engine)
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()
