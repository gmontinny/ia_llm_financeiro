from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.config import GROQ_API_KEY, GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE, EMBEDDING_MODEL


def get_llm(provider: str = "groq"):
    if provider == "gemini":
        return GoogleGenAI(model="models/gemini-2.5-flash")
    return Groq(model=MODEL_NAME, temperature=TEMPERATURE)


def get_embed_model():
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def configure_settings(llm, embed_model=None):
    Settings.llm = llm
    if embed_model:
        Settings.embed_model = embed_model
