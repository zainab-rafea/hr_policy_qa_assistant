"""Load HR policy documents and build a FAISS vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DEFAULT_POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_policy_documents(policies_dir: Path | str = DEFAULT_POLICIES_DIR) -> list:
    """Load markdown policy files and split by headers, then by chunk size."""
    policies_dir = Path(policies_dir)
    loader = DirectoryLoader(
        str(policies_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    raw_docs = loader.load()

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "policy"),
            ("##", "section"),
        ],
        strip_headers=False,
    )
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )

    documents = []
    for doc in raw_docs:
        source_name = Path(doc.metadata.get("source", "unknown")).stem.replace("_", " ").title()
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata["source"] = doc.metadata.get("source", "")
            section.metadata["policy_name"] = section.metadata.get("policy") or source_name
            if "section" in section.metadata:
                section.metadata["section_label"] = section.metadata["section"]
            else:
                section.metadata["section_label"] = section.metadata["policy_name"]
        documents.extend(chunk_splitter.split_documents(sections) if sections else [doc])

    # Fallback: if header split produced nothing useful, chunk raw docs
    if not documents:
        documents = chunk_splitter.split_documents(raw_docs)
        for d in documents:
            d.metadata["policy_name"] = Path(d.metadata.get("source", "policy")).stem
            d.metadata["section_label"] = d.metadata["policy_name"]

    return documents


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    policies_dir: Path | str = DEFAULT_POLICIES_DIR,
    persist_dir: Path | str | None = DEFAULT_INDEX_DIR,
) -> FAISS:
    """Create (and optionally persist) a FAISS index over policy chunks."""
    documents = load_policy_documents(policies_dir)
    if not documents:
        raise FileNotFoundError(f"No policy documents found in {policies_dir}")

    vectorstore = FAISS.from_documents(documents, get_embeddings())
    if persist_dir is not None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(persist_dir))
    return vectorstore


def load_or_build_vectorstore(
    policies_dir: Path | str = DEFAULT_POLICIES_DIR,
    persist_dir: Path | str = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> FAISS:
    persist_dir = Path(persist_dir)
    index_file = persist_dir / "index.faiss"
    if not rebuild and index_file.exists():
        return FAISS.load_local(
            str(persist_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return build_vectorstore(policies_dir=policies_dir, persist_dir=persist_dir)
