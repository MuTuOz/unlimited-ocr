"""RAG phase: FAISS index over the OCR output + a chat model on top.

This phase never touches the GPU or the OCR model. It reads markdown from
disk, so you can run it on a laptop while the OCR box does something else.

Page-level Documents with a `page` in metadata, so an answer can cite which
page it came from. Concatenating the whole document into one blob throws that
away, and it is one of the main reasons to bother with retrieval at all.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ocr import strip_layout_tags

log = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile")

PAGE_NUM = re.compile(r"page_(\d+)")

PROMPT = PromptTemplate.from_template(
    "You are reading OCR output from a scanned document. Tables are given as "
    "HTML. Answer using only the context below. If the context does not "
    "contain the answer, say so plainly rather than guessing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def load_pages(ocr_dir: Path) -> list[Document]:
    """One Document per page file, layout tags stripped."""
    docs: list[Document] = []
    for path in sorted(ocr_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_layout_tags(raw)
        if not text:
            log.warning("%s is empty, skipping", path.name)
            continue

        match = PAGE_NUM.search(path.stem)
        page = int(match.group(1)) if match else None
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page},
            )
        )

    if not docs:
        raise RuntimeError(
            f"No usable text in {ocr_dir}. Run the ocr command first, and "
            f"check that the OCR output is not blank."
        )

    log.info("Loaded %d pages from %s", len(docs), ocr_dir)
    return docs


def build_index(ocr_dir: Path, index_dir: Path, chunk_size: int = 1200) -> int:
    """Chunk, embed, persist. Returns the number of chunks."""
    docs = load_pages(ocr_dir)

    # Separators favour markdown structure so tables survive intact where
    # possible -- a table split across two chunks retrieves badly.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n</table>", "\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(docs)
    log.info("%d pages -> %d chunks", len(docs), len(splits))

    store = FAISS.from_documents(splits, _embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    log.info("Index saved to %s", index_dir)
    return len(splits)


def _embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def load_index(index_dir: Path) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise RuntimeError(f"No index at {index_dir}. Run the index command first.")
    # Safe here: we wrote this file ourselves in the previous step.
    return FAISS.load_local(
        str(index_dir), _embeddings(), allow_dangerous_deserialization=True
    )


def _chat_model():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Pass it with -e GROQ_API_KEY on docker "
            "run, or put it in .env for docker compose."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model=CHAT_MODEL, temperature=0)


def make_chain(index_dir: Path, k: int = 4) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=_chat_model(),
        retriever=load_index(index_dir).as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )


def ask(chain: RetrievalQA, question: str) -> tuple[str, list[int]]:
    result = chain.invoke({"query": question})
    pages = sorted(
        {
            d.metadata.get("page")
            for d in result.get("source_documents", [])
            if d.metadata.get("page") is not None
        }
    )
    return result["result"], pages


def dump(ocr_dir: Path) -> str:
    """No retrieval, no LLM, no API key: just the OCR text.

    For a single page or a short document this is usually what you actually
    want. Retrieval only starts paying off when the text no longer fits in a
    context window.
    """
    return "\n\n".join(d.page_content for d in load_pages(ocr_dir))
