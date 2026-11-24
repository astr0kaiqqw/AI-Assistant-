import logging
import os
import sys
from hashlib import sha256
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import faiss
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from app.document_text import (
    MAX_UPLOAD_BYTES, DocumentInputError, document_name, extract_text,
)
from app.models import Document, DocumentChunk, DocumentIndex
from app.schemas import DocumentHit


WORKSPACE = "local"
DEFAULT_STORAGE = Path(__file__).resolve().parent.parent / ".rag-data"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
CHUNK_TOKENS = 110
OVERLAP_TOKENS = 24
MAX_CHUNKS = 5_000
logger = logging.getLogger(__name__)


class DocumentNotFound(Exception):
    pass


class DocumentSearchUnavailable(Exception):
    pass


class DocumentStore:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions
        self.directory = Path(os.getenv("RAG_STORAGE_DIR") or DEFAULT_STORAGE)
        self._model = None
        self._model_lock = Lock()

    def _embedding_model(self):
        # Call under _model_lock to share one CPU model across request threads.
        if self._model is None:
            if sys.platform == "darwin":
                import torch

                # Batched CPU inference can crash in multithreaded PyTorch on macOS.
                torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        return self._model

    def _chunks_and_embeddings(self, pages):
        chunks = []
        try:
            with self._model_lock:
                model = self._embedding_model()
                for page, value in pages:
                    offsets = model.tokenizer(
                        value, return_offsets_mapping=True, add_special_tokens=False, verbose=False,
                    )["offset_mapping"]
                    for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS):
                        end = min(start + CHUNK_TOKENS, len(offsets))
                        chunk = value[offsets[start][0]:offsets[end - 1][1]].strip()
                        if chunk:
                            chunks.append((page, chunk))
                            if len(chunks) > MAX_CHUNKS:
                                raise DocumentInputError("The document has too many indexable chunks", 413)
                        if end == len(offsets):
                            break
                if not chunks:
                    raise DocumentInputError("The document has no indexable text")
                embeddings = model.encode(
                    [chunk for _, chunk in chunks], normalize_embeddings=True, show_progress_bar=False,
                )
        except DocumentInputError:
            raise
        except Exception:
            raise DocumentSearchUnavailable from None
        return chunks, embeddings

    def _query_embedding(self, query: str):
        try:
            with self._model_lock:
                model = self._embedding_model()
                return model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            raise DocumentSearchUnavailable from None

    def _document(self, session: Session, identifier: UUID) -> Document:
        document = session.get(Document, identifier)
        if document is None or document.workspace != WORKSPACE:
            raise DocumentNotFound
        return document

    def _check_duplicate(self, session: Session, digest: str) -> None:
        existing = session.scalar(select(Document.id).where(
            Document.workspace == WORKSPACE, Document.sha256 == digest,
        ))
        if existing is not None:
            raise DocumentInputError("This file is already indexed in the workspace", 409)

    def has_documents(self) -> bool:
        with self.sessions() as session:
            return session.scalar(select(DocumentIndex.workspace).where(
                DocumentIndex.workspace == WORKSPACE
            )) is not None

    def upload(self, filename: str, data: bytes) -> Document:
        if len(data) > MAX_UPLOAD_BYTES:
            raise DocumentInputError("Files are limited to 10 MiB", 413)
        name, format = document_name(filename)
        digest = sha256(data).hexdigest()
        with self.sessions() as session:
            self._check_duplicate(session, digest)
        pages, page_count = extract_text(data, format)
        chunks, embeddings = self._chunks_and_embeddings(pages)

        identifier = uuid4()
        source = self.directory / f"{identifier}.{format}"
        created_file = False
        try:
            with self.sessions.begin() as session:
                # Serialize index updates across requests and app processes.
                session.execute(text("BEGIN IMMEDIATE"))
                self._check_duplicate(session, digest)
                document = Document(
                    id=identifier, workspace=WORKSPACE, name=name, format=format,
                    sha256=digest, size_bytes=len(data), page_count=page_count, chunk_count=len(chunks),
                )
                rows = [DocumentChunk(document_id=identifier, page=page, text=chunk) for page, chunk in chunks]
                session.add(document)
                session.flush()
                session.add_all(rows)
                session.flush()

                snapshot = session.get(DocumentIndex, WORKSPACE)
                if snapshot is None:
                    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
                    snapshot = DocumentIndex(workspace=WORKSPACE)
                    session.add(snapshot)
                else:
                    index = self._read_index(snapshot.data)
                index.add_with_ids(embeddings, np.array([row.id for row in rows], dtype=np.int64))
                snapshot.data = faiss.serialize_index(index).tobytes()

                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                with source.open("xb") as file:
                    created_file = True
                    file.write(data)
            return document
        except BaseException:
            if created_file:
                self._remove_file(source)
            raise

    def list_documents(self, limit: int, offset: int) -> list[Document]:
        query = (
            select(Document).where(Document.workspace == WORKSPACE)
            .order_by(Document.created_at.desc(), Document.id.desc()).limit(limit).offset(offset)
        )
        with self.sessions() as session:
            return list(session.scalars(query))

    def get(self, identifier: UUID) -> Document:
        with self.sessions() as session:
            return self._document(session, identifier)

    def delete(self, identifier: UUID) -> None:
        with self.sessions.begin() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            document = self._document(session, identifier)
            ids = list(session.scalars(
                select(DocumentChunk.id).where(DocumentChunk.document_id == identifier)
            ))
            snapshot = session.get(DocumentIndex, WORKSPACE)
            if snapshot is not None:
                index = self._read_index(snapshot.data)
                index.remove_ids(np.array(ids, dtype=np.int64))
                if index.ntotal:
                    snapshot.data = faiss.serialize_index(index).tobytes()
                else:
                    session.delete(snapshot)
            session.delete(document)
        self._remove_file(self.directory / f"{identifier}.{document.format}")

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove a local document file: %s", path.name)

    @staticmethod
    def _read_index(data: bytes):
        try:
            return faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        except RuntimeError:
            raise DocumentSearchUnavailable from None

    def search(self, query: str, top_k: int) -> list[DocumentHit]:
        with self.sessions() as session:
            snapshot = session.get(DocumentIndex, WORKSPACE)
            if snapshot is None:
                return []
        index = self._read_index(snapshot.data)
        scores, ids = index.search(self._query_embedding(query), top_k)
        ranked = [(int(key), float(score)) for key, score in zip(ids[0], scores[0]) if key >= 0]
        if not ranked:
            return []
        with self.sessions() as session:
            rows = session.execute(
                select(DocumentChunk, Document).join(Document)
                .where(Document.workspace == WORKSPACE, DocumentChunk.id.in_([key for key, _ in ranked]))
            )
            matched = {chunk.id: (chunk, document) for chunk, document in rows}

        hits = []
        for key, score in ranked:
            if key in matched:
                chunk, document = matched[key]
                hits.append(DocumentHit(
                    chunk_id=key, document_id=document.id, document_name=document.name,
                    page=chunk.page, text=chunk.text, score=score,
                ))
        return hits
