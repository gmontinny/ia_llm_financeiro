import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

SUMMARY_TEMPLATE = """
Você é um analista financeiro experiente.
Ao ler o relatório a seguir, extraia insights financeiros relevantes e explique-os de forma clara e didática, como se estivesse apresentando para gestores não especialistas.
Utilize linguagem acessível e destaque pontos importantes sobre lucros, despesas, fluxo de caixa, riscos e oportunidades.
Retorne o texto em linguagem natural e com caracteres que possuem em um teclado comum, sem caracteres ou símbolos LaTex.
Resuma de forma breve e objetiva. Retorne a mensagem direto, sem apresentações no início.

Conteúdo do documento:
'{}'
"""

ANALYSIS_TEMPLATE = """
Você é um analista financeiro experiente.
Ao ler o relatório a seguir, extraia insights financeiros relevantes e explique-os de forma clara e didática, como se estivesse apresentando para gestores não especialistas.
Utilize linguagem acessível e destaque pontos importantes sobre lucros, despesas, fluxo de caixa, riscos e oportunidades.

Responda à pergunta:
'{}'

Conteúdo do documento:
'{}'
"""

AGENT_SYSTEM_PROMPT = """
You are a financial analyst assistant. You already have access to the user's uploaded document through your tools.
ALWAYS use your available tools to answer questions - never say you don't have access to the data.
For any question about finances, revenues, expenses, analysis or hypotheses, use the appropriate tool to query the data first, then provide your analysis.
"""

MAX_CONTENT_CHARS = 200000
MAX_CSV_ROWS = 500
SIMILARITY_TOP_K = 3
