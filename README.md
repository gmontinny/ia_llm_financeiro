# 💵 Assistente Financeiro com LLMs

Assistente de análise financeira que utiliza LLMs (Groq/Gemini) com LlamaIndex para processar documentos financeiros (PDF e CSV), gerar insights automáticos e responder perguntas via agente ReAct com RAG.

## Tecnologias

- **LLMs**: Groq (Llama 3.3 70B) e Google Gemini 2.5 Flash
- **Framework**: LlamaIndex (agentes, RAG, query engines)
- **Embeddings**: HuggingFace (BAAI/bge-small-en-v1.5)
- **Interface**: Streamlit
- **Tradução**: deep-translator (Google Translator)

## Estrutura do Projeto

```
ia_llm_financeiro/
├── src/
│   ├── __init__.py
│   ├── config.py            # Variáveis de ambiente, templates de prompt, system prompt do agente e constantes
│   ├── llm.py               # Inicialização dos modelos LLM e embeddings
│   ├── document_loader.py   # Leitura de PDFs/CSVs e indexação vetorial
│   ├── tools.py             # Ferramentas do agente (query engines com descrições contextuais, cálculos)
│   ├── agent.py             # Criação do agente com system prompt e execução assíncrona
│   └── utils.py             # Formatação de respostas, tradução e sumarização
├── docs/
│   ├── fluxo_caixa.csv      # Exemplo: fluxo de caixa (Jan-Mar 2025)
│   └── relatorio-financeiro.pdf  # Exemplo: relatório anual NexusEdge Solutions
├── notebook/
│   └── LLMs para empresas e negócios - Finanças.ipynb  # Notebook original de referência
├── app.py                   # Interface Streamlit (com cache de embeddings)
├── requirements.txt
├── .env                     # Chaves de API (não versionado)
├── .env.example             # Modelo de configuração
├── .gitignore
└── README.md
```

## Configuração

1. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

2. Copie o arquivo de exemplo e configure suas chaves de API:
```bash
cp .env.example .env
```

Edite o `.env`:
```
GROQ_API_KEY=sua_chave_groq
GOOGLE_API_KEY=sua_chave_google
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.3
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

## Execução

```bash
streamlit run app.py
```

Acesse em: http://localhost:8501

## Funcionalidades

- **Upload de documentos**: PDF ou CSV pela barra lateral
- **Sumarização automática**: insights financeiros gerados ao enviar o documento
- **Chat com o documento**: perguntas em linguagem natural via agente com RAG
- **Análise de planilhas**: consultas inteligentes em CSVs via PandasQueryEngine
- **Ferramentas de cálculo**: soma, subtração, multiplicação, divisão e taxa de crescimento
- **Tradução automática**: PT → EN → PT para melhor precisão com documentos em inglês
- **Seleção de modelo**: escolha entre Gemini e Groq pela sidebar
- **Log do agente**: visualização das etapas de raciocínio para transparência

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `config.py` | Variáveis de ambiente, templates de sumarização/análise, system prompt do agente e constantes (limites, top_k) |
| `llm.py` | Fábrica de LLMs (Groq/Gemini) e modelo de embeddings HuggingFace |
| `document_loader.py` | Leitura de PDFs com SimpleDirectoryReader, CSVs com Pandas, e indexação vetorial |
| `tools.py` | QueryEngineTool (RAG para PDFs), PandasQueryEngine (CSV) com descrições contextuais, e FunctionTools de cálculo |
| `agent.py` | Criação do FunctionAgent com system prompt que garante uso das tools, e execução assíncrona com captura de logs |
| `utils.py` | Formatação de respostas (remoção de tags think), tradução PT↔EN e sumarização via LLM |
| `app.py` | Interface Streamlit com cache de embeddings (`@st.cache_resource`), upload, chat e exibição de dados |

## Arquitetura

```
Upload (PDF/CSV)
       │
       ▼
┌─────────────────┐
│ document_loader  │ → Leitura + Indexação vetorial (VectorStoreIndex)
└────────┬────────┘   ou PandasQueryEngine (CSV)
         │
         ▼
┌─────────────────┐
│     tools        │ → QueryEngineTool (RAG) + FunctionTools (cálculos)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     agent        │ → FunctionAgent com system prompt + LLM escolhido
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    app.py        │ → Interface Streamlit (chat + resumo + dados)
└─────────────────┘
```
