import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from src.llm import get_embed_model
from src.config import MAX_CONTENT_CHARS, MAX_CSV_ROWS


def load_and_index(file_path: str):
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    embed_model = get_embed_model()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    return docs, index


def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def extract_text_from_docs(docs, max_chars: int = MAX_CONTENT_CHARS) -> str:
    content = "\n".join([doc.text for doc in docs])
    return content[:max_chars]


def extract_text_from_df(df: pd.DataFrame, max_rows: int = MAX_CSV_ROWS) -> str:
    return df.head(max_rows).to_string()
