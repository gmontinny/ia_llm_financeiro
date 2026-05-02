# Assistente Financeiro Inteligente: Unindo LLMs e RAG para Análise de Dados

## 1. Introdução

No cenário financeiro atual, gestores lidam diariamente com relatórios extensos em PDF, planilhas de fluxo de caixa e dezenas de indicadores que precisam ser cruzados para embasar decisões estratégicas. Esse processo manual é lento, propenso a erros e exige conhecimento técnico que nem sempre está disponível em todas as camadas da organização.

O projeto **Assistente Financeiro com LLMs** surge para resolver esse gargalo: uma aplicação que permite ao usuário **fazer upload de um documento financeiro** (PDF ou CSV) e imediatamente **conversar com os dados em linguagem natural**, obtendo insights, cálculos e análises sem escrever uma linha de código ou fórmula.

A solução combina três pilares tecnológicos:
- **LLMs (Large Language Models)** para compreensão e geração de linguagem natural
- **RAG (Retrieval-Augmented Generation)** para fundamentar respostas em dados reais do documento
- **Agentes autônomos** que decidem qual ferramenta usar para cada tipo de pergunta

---

## 2. O Problema

Departamentos financeiros enfrentam desafios recorrentes:

- **Volume de dados**: relatórios anuais com dezenas de páginas e planilhas com centenas de linhas de transações
- **Barreira técnica**: para extrair insights de um CSV, é necessário conhecer SQL, Pandas ou Excel avançado
- **Tempo de resposta**: uma análise trimestral que poderia levar horas pode ser respondida em segundos com a abordagem correta
- **Acessibilidade**: gestores não técnicos dependem de analistas para obter respostas simples como "qual foi o maior gasto em março?"

A proposta é democratizar o acesso à análise financeira através de uma interface conversacional inteligente.

---

## 3. Tecnologias Utilizadas

### 3.1 Modelos de Linguagem (LLMs)

O sistema oferece dois provedores, selecionáveis pela interface:

| Modelo | Provedor | Destaque |
|---|---|---|
| Llama 3.3 70B | Groq | Inferência ultrarrápida via hardware especializado (LPU) |
| Gemini 2.5 Flash | Google | Grande janela de contexto (~1M tokens), ideal para documentos extensos |

A escolha entre os modelos é feita em tempo de execução, sem necessidade de reprocessar o documento:

```python
def get_llm(provider: str = "groq"):
    if provider == "gemini":
        return GoogleGenAI(model="models/gemini-2.5-flash")
    return Groq(model=MODEL_NAME, temperature=TEMPERATURE)
```

### 3.2 Framework LlamaIndex

O **LlamaIndex** é a espinha dorsal do projeto, responsável por toda a orquestração entre dados, embeddings, ferramentas e o agente. Os componentes utilizados incluem:

- **SimpleDirectoryReader**: leitura e parsing de PDFs
- **VectorStoreIndex**: indexação vetorial para busca semântica
- **QueryEngineTool**: ferramenta de RAG que o agente usa para consultar PDFs
- **PandasQueryEngine**: traduz perguntas em linguagem natural para código Pandas executável
- **FunctionAgent**: agente que decide autonomamente qual ferramenta usar

### 3.3 Embeddings e Busca Vetorial

O modelo **BAAI/bge-small-en-v1.5** do HuggingFace converte textos em vetores de 384 dimensões, permitindo busca por similaridade semântica. Foi escolhido por ser leve (~130MB) e eficiente para tarefas de recuperação de informação.

O processo de RAG funciona assim:
1. O texto do PDF é fragmentado em chunks
2. Cada chunk é convertido em um vetor pelo modelo de embedding
3. Quando o usuário faz uma pergunta, ela também é vetorizada
4. Os chunks mais similares são recuperados (top_k=3) e enviados como contexto para a LLM

Para evitar recarregamento a cada interação no Streamlit, o modelo é cacheado:

```python
@st.cache_resource
def cached_embed_model():
    return get_embed_model()
```

### 3.4 Interface e Tradução

- **Streamlit**: transforma o projeto em uma aplicação web interativa com upload, chat, exibição de dados tabulares e logs do agente
- **deep-translator (Google Translator)**: implementa o pipeline de tradução PT → EN → PT, útil quando o documento está em inglês mas o usuário pergunta em português

---

## 4. Arquitetura do Sistema

### 4.1 Visão Geral

```
Upload (PDF/CSV)
       │
       ▼
┌─────────────────┐
│ document_loader  │ → Leitura + Indexação vetorial (VectorStoreIndex)
└────────┬────────┘   ou carregamento em DataFrame (CSV)
         │
         ▼
┌─────────────────┐
│     tools        │ → QueryEngineTool (RAG) + PandasQueryEngine + FunctionTools
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     agent        │ → FunctionAgent com system prompt + LLM escolhido
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    app.py        │ → Interface Streamlit (chat + resumo + dados + logs)
└─────────────────┘
```

### 4.2 Fluxo Detalhado por Tipo de Arquivo

**Para PDFs:**
1. O `SimpleDirectoryReader` extrai o texto de cada página
2. O `VectorStoreIndex` fragmenta e indexa o conteúdo usando embeddings
3. Uma `QueryEngineTool` é criada, permitindo ao agente buscar trechos relevantes por similaridade semântica
4. A LLM recebe os trechos recuperados como contexto e gera a resposta

**Para CSVs:**
1. O arquivo é carregado em um `DataFrame` do Pandas com parsing automático de datas
2. Um `PandasQueryEngine` é criado — ele traduz perguntas em linguagem natural para código Pandas
3. O código é executado sobre o DataFrame real, garantindo precisão numérica
4. O resultado da execução é retornado ao agente, que formula a resposta final

### 4.3 O Agente e suas Ferramentas

O `FunctionAgent` é o componente central de decisão. Ao receber uma pergunta, ele analisa a intenção e seleciona a ferramenta mais adequada:

| Ferramenta | Tipo | Uso |
|---|---|---|
| `doc_search` | QueryEngineTool | Busca semântica em PDFs via RAG |
| `query_spreadsheet` | FunctionTool | Consultas em CSVs via Pandas |
| `add`, `subtract`, `multiply`, `divide` | FunctionTool | Operações matemáticas básicas |
| `growth_rate` | FunctionTool | Cálculo de taxa de crescimento percentual |

O agente pode encadear múltiplas ferramentas em uma única resposta. Por exemplo, para "calcule a taxa de crescimento da receita de janeiro para março", ele pode:
1. Usar `query_spreadsheet` para obter a receita de janeiro
2. Usar `query_spreadsheet` novamente para obter a receita de março
3. Usar `growth_rate` para calcular o percentual

---

## 5. Engenharia de Prompts

Um dos aspectos mais críticos do projeto é a construção dos prompts que guiam o comportamento da LLM.

### 5.1 System Prompt do Agente

O agente recebe um system prompt que garante que ele sempre utilize as ferramentas disponíveis ao invés de alegar falta de acesso aos dados:

```
You are a financial analyst assistant. You already have access to the
user's uploaded document through your tools.
ALWAYS use your available tools to answer questions - never say you
don't have access to the data.
For any question about finances, revenues, expenses, analysis or
hypotheses, use the appropriate tool to query the data first, then
provide your analysis.
```

Sem esse prompt, o agente frequentemente respondia "preciso que você forneça o arquivo" mesmo com os dados já carregados — um problema comum em agentes LLM que não recebem instruções claras sobre o contexto disponível.

### 5.2 Descrições Contextuais das Ferramentas

A descrição de cada ferramenta é fundamental para que o agente saiba quando usá-la. A ferramenta de CSV, por exemplo, tem uma descrição rica:

```python
"""Queries the already loaded CSV spreadsheet with financial data
(revenues, expenses, dates, categories). Use this tool for ANY question
about the data, finances, revenues, expenses, hypotheses or analysis.
The data is already available - just pass the user's question."""
```

Descrições genéricas como "Useful for querying data" levavam o agente a ignorar a ferramenta em perguntas mais abstratas como "qual sua hipótese sobre receitas e despesas?".

### 5.3 Template de Sumarização

Ao receber um documento, o sistema gera automaticamente um resumo usando um prompt que simula um analista financeiro:

```
Você é um analista financeiro experiente.
Ao ler o relatório a seguir, extraia insights financeiros relevantes
e explique-os de forma clara e didática, como se estivesse apresentando
para gestores não especialistas.
Utilize linguagem acessível e destaque pontos importantes sobre lucros,
despesas, fluxo de caixa, riscos e oportunidades.
```

### 5.4 Tratamento de Tags de Raciocínio

Modelos como o DeepSeek R1 (via Groq) retornam tags `<think>...</think>` com o raciocínio interno. O sistema trata essas tags de duas formas:
- **Na resposta final**: remove as tags, exibindo apenas a conclusão
- **Nos logs do agente**: converte para `[pensando...]`, permitindo ao usuário acompanhar o raciocínio

```python
def format_response(res: str, return_thinking: bool = False) -> str:
    res = res.strip()
    if return_thinking:
        res = res.replace("<think>", "[pensando...] ")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    return res
```

---

## 6. Desafios Técnicos e Soluções

### 6.1 Cadeia de Imports do llama-index-experimental

O pacote `llama-index-experimental` possui um `__init__.py` que importa módulos de finetuning e nudge, que por sua vez dependem de `mistralai`. Isso causava `ImportError` mesmo sem usar esses módulos.

**Solução**: lazy import do `PandasQueryEngine` dentro da função que o utiliza, combinado com a instalação de `mistralai>=1.0,<2.0` (a versão 2.x mudou a API e removeu a classe `Mistral`).

### 6.2 Modelo de Embedding Pesado

O modelo `BAAI/bge-m3` (1.7GB) causava erro de meta tensor (`Cannot copy out of meta tensor`) ao ser carregado repetidamente pelo Streamlit.

**Solução**: troca para `BAAI/bge-small-en-v1.5` (130MB) com cache via `@st.cache_resource`, garantindo carregamento único.

### 6.3 Agente Sem Contexto entre Perguntas

O Streamlit faz rerun completo do script a cada interação. O agente perdia o contexto e dizia não ter acesso aos dados.

**Solução**: combinação de `st.session_state` para persistir o agente entre reruns, system prompt explícito e descrições ricas nas ferramentas.

### 6.4 Execução Assíncrona no Streamlit

O LlamaIndex moderno usa `async/await` para execução de agentes, mas o Streamlit roda de forma síncrona.

**Solução**: uso de `asyncio.run()` para executar a função assíncrona do agente, com captura de stdout para coletar os logs de raciocínio:

```python
async def run_agent(agent, query: str):
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        handler = agent.run(query)
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                print(f"Call {event.tool_name} with args {event.tool_kwargs}")
            elif isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
        response = await handler
        formatted = format_response(str(response), return_thinking=True)
    finally:
        sys.stdout = old_stdout
    return formatted, captured.getvalue()
```

---

## 7. Exemplos de Aplicação

### Cenário 1: Análise de Relatório Anual (PDF)

**Pergunta:** "Quais foram os principais riscos mencionados no relatório?"

**O que acontece internamente:**
1. O agente identifica que precisa buscar no documento
2. Aciona a ferramenta `doc_search` com a query
3. O `VectorStoreIndex` recupera os 3 chunks mais relevantes por similaridade
4. A LLM recebe os chunks como contexto e gera um resumo dos riscos

### Cenário 2: Gestão de Fluxo de Caixa (CSV)

**Pergunta:** "Quais os maiores gastos em março?"

**O que acontece internamente:**
1. O agente aciona `query_spreadsheet`
2. O `PandasQueryEngine` gera o código:
   ```python
   df[(df['date'].dt.month == 3) & (df['type'] == 'Despesa')].sort_values('value', ascending=False)
   ```
3. O código é executado sobre o DataFrame real
4. O agente formata o resultado: "Folha de Pagamento (R$75.200), Infraestrutura Cloud (R$35.200), Marketing (R$23.500)"

### Cenário 3: Cálculos Compostos

**Pergunta:** "Compare as despesas de janeiro e fevereiro e calcule a taxa de crescimento"

**O que acontece internamente:**
1. O agente usa `query_spreadsheet` para obter o total de despesas de janeiro
2. Usa `query_spreadsheet` novamente para fevereiro
3. Aciona `growth_rate(previous=159200, current=170900)` → retorna 7.35%
4. Formula a resposta com os valores e o percentual calculado

---

## 8. Estrutura do Projeto

```
ia_llm_financeiro/
├── src/
│   ├── config.py            # Variáveis de ambiente, templates de prompt, system prompt e constantes
│   ├── llm.py               # Fábrica de LLMs (Groq/Gemini) e modelo de embeddings
│   ├── document_loader.py   # Leitura de PDFs/CSVs e indexação vetorial
│   ├── tools.py             # QueryEngineTool, PandasQueryEngine e FunctionTools de cálculo
│   ├── agent.py             # Criação do FunctionAgent e execução assíncrona com captura de logs
│   └── utils.py             # Formatação de respostas, tradução PT↔EN e sumarização
├── docs/                    # Documentos de exemplo (relatório PDF + fluxo de caixa CSV)
├── notebook/                # Notebook original de referência (Colab)
├── app.py                   # Interface Streamlit com cache de embeddings
├── requirements.txt
├── .env.example
└── README.md
```

A separação modular segue o princípio de responsabilidade única: cada módulo tem uma função clara e pode ser testado ou substituído independentemente.

---

## 9. Possíveis Melhorias Futuras

- **Memória conversacional**: manter histórico de perguntas e respostas entre interações usando `Context` do LlamaIndex
- **Suporte a múltiplos documentos**: permitir upload simultâneo de vários arquivos e cruzamento de dados entre eles
- **Geração de gráficos**: integrar bibliotecas como Plotly ou Matplotlib para visualizações automáticas a partir dos dados
- **Saída estruturada**: retornar respostas em formato JSON ou tabelas Markdown para integração com outros sistemas
- **Deploy em nuvem**: containerizar com Docker e hospedar em serviços como AWS ECS ou Google Cloud Run
- **Avaliação de qualidade**: implementar métricas de avaliação (faithfulness, relevancy) usando frameworks como RAGAS

---

## 10. Conclusão

O projeto demonstra como a combinação de LLMs, RAG e agentes autônomos pode transformar a análise financeira de um processo manual e técnico em uma conversa natural e acessível. A capacidade de "conversar com os dados" não é apenas uma conveniência — é uma mudança de paradigma na forma como organizações interagem com informação corporativa.

Os principais aprendizados técnicos incluem a importância da engenharia de prompts (tanto no system prompt do agente quanto nas descrições das ferramentas), o cuidado com gerenciamento de estado em aplicações Streamlit, e as nuances de integração entre bibliotecas em rápida evolução como o LlamaIndex.

---

## Referências

- **LlamaIndex Documentation:** [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
- **Groq Cloud API:** [https://console.groq.com/](https://console.groq.com/)
- **Google Gemini API:** [https://aistudio.google.com/](https://aistudio.google.com/)
- **Streamlit Docs:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **HuggingFace - BGE Small:** [https://huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **RAG (Lewis et al., 2020):** [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- **ReAct (Yao et al., 2022):** [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
