from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile

from app.document_text import MAX_UPLOAD_BYTES, DocumentInputError
from app.schemas import DocumentHit, DocumentResponse


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=201, response_model=DocumentResponse)
def upload_document(request: Request, file: Annotated[UploadFile, File()]):
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise DocumentInputError("Files are limited to 10 MiB", 413)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    return request.app.state.documents.upload(file.filename or "", data)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
):
    return request.app.state.documents.list_documents(limit, offset)


@router.get("/search", response_model=list[DocumentHit])
def search_documents(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=300)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 4,
):
    if not q.strip():
        raise HTTPException(status_code=422, detail="Search query must contain text")
    return request.app.state.documents.search(q.strip(), top_k)


@router.get("/{document_id}", response_model=DocumentResponse)
def inspect_document(document_id: UUID, request: Request):
    return request.app.state.documents.get(document_id)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: UUID, request: Request):
    request.app.state.documents.delete(document_id)
    return Response(status_code=204)
