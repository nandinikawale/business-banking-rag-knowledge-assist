from __future__ import annotations

import re
from pathlib import Path

from banking_rag.models import Chunk, KnowledgeDocument

REQUIRED_METADATA = {"document_id", "title", "category", "version", "effective_date"}


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("Knowledge document must start with Markdown frontmatter")
    _, raw_metadata, body = text.split("---", 2)
    metadata: dict[str, str] = {}
    for line in raw_metadata.strip().splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid metadata line: {line}")
        metadata[key.strip()] = value.strip().strip('"')
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"Missing metadata fields: {sorted(missing)}")
    return metadata, body.strip()


def load_documents(knowledge_dir: Path) -> list[KnowledgeDocument]:
    documents: list[KnowledgeDocument] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        metadata, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        documents.append(
            KnowledgeDocument(
                **metadata,
                text=body,
                source_path=path.relative_to(knowledge_dir.parent.parent).as_posix(),
            )
        )
    if not documents:
        raise FileNotFoundError(f"No Markdown documents found in {knowledge_dir}")
    return documents


def chunk_document(document: KnowledgeDocument, max_characters: int = 1200) -> list[Chunk]:
    sections = re.split(r"(?m)^##\s+", document.text)
    chunks: list[Chunk] = []
    sequence = 0
    for raw_section in sections:
        raw_section = raw_section.strip()
        if not raw_section:
            continue
        first_line, _, remainder = raw_section.partition("\n")
        section = first_line.strip("# ")
        content = remainder.strip() or first_line
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip()
            if buffer and len(candidate) > max_characters:
                sequence += 1
                chunks.append(_make_chunk(document, section, buffer, sequence))
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            sequence += 1
            chunks.append(_make_chunk(document, section, buffer, sequence))
    return chunks


def _make_chunk(document: KnowledgeDocument, section: str, text: str, sequence: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document.document_id}-c{sequence:03d}",
        document_id=document.document_id,
        title=document.title,
        category=document.category,
        section=section,
        version=document.version,
        effective_date=document.effective_date,
        text=text,
        source_path=document.source_path,
    )


def load_and_chunk(knowledge_dir: Path) -> list[Chunk]:
    return [
        chunk for document in load_documents(knowledge_dir) for chunk in chunk_document(document)
    ]
