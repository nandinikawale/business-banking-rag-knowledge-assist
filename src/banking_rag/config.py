from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    vector_dir: Path = PROJECT_ROOT / ".chromadb"
    collection_name: str = "fictional_business_banking_knowledge"
    embedding_dimensions: int = 384
    default_top_k: int = 4


settings = Settings()
