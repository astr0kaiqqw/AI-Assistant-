from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 250
MAX_TEXT_CHARS = 1_000_000


class DocumentInputError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def document_name(filename: str) -> tuple[str, str]:
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    if not name or len(name) > 255 or not name.isprintable():
        raise DocumentInputError("Invalid document filename")
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise DocumentInputError("Only PDF and UTF-8 TXT files are supported", 415)
    return name, suffix[1:]


def extract_text(data: bytes, format: str) -> tuple[list[tuple[int | None, str]], int | None]:
    if not data:
        raise DocumentInputError("The document is empty")
    if format == "txt":
        try:
            value = data.decode("utf-8-sig")
        except UnicodeError:
            raise DocumentInputError("TXT files must use UTF-8 encoding") from None
        if any(not char.isprintable() and not char.isspace() for char in value):
            raise DocumentInputError("The TXT file contains binary data")
        if len(value) > MAX_TEXT_CHARS:
            raise DocumentInputError("The extracted text exceeds the workspace limit", 413)
        pages, page_count = [(None, value.strip())], None
    else:
        if not data.startswith(b"%PDF-"):
            raise DocumentInputError("The file is not a valid PDF")
        pages, page_count = extract_pdf(data)

    pages = [(page, value) for page, value in pages if value]
    if not pages:
        raise DocumentInputError("The document has no extractable text")
    return pages, page_count


def extract_pdf(data: bytes) -> tuple[list[tuple[int | None, str]], int]:
    pages = []
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise DocumentInputError("Encrypted PDFs are unsupported")
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            raise DocumentInputError(f"PDFs are limited to {MAX_PDF_PAGES} pages", 413)
        total_chars = 0
        for number, page in enumerate(reader.pages, start=1):
            value = page.extract_text().strip()
            total_chars += len(value)
            if total_chars > MAX_TEXT_CHARS:
                raise DocumentInputError("The extracted text exceeds the workspace limit", 413)
            pages.append((number, value))
    except (PdfReadError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        raise DocumentInputError("The file is not a readable PDF") from None
    return pages, page_count
