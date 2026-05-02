import asyncio
import tempfile
import streamlit as st
from llama_index.core import Settings

from src.llm import get_llm, get_embed_model, configure_settings
from src.document_loader import load_and_index, load_csv, extract_text_from_docs, extract_text_from_df
from src.tools import create_query_engine_tool, create_pandas_tool, get_math_tools
from src.agent import run_agent, create_agent
from src.utils import translate, summarize_doc

# --- Page Config ---
st.set_page_config(page_title="Análise de Documentos Financeiros", page_icon="💵", layout="wide")
st.title("Análise de Documentos Financeiros 💵")

# --- Session State ---
for key in ["docs_list", "agent", "summary", "df"]:
    if key not in st.session_state:
        st.session_state[key] = None


@st.cache_resource
def cached_embed_model():
    return get_embed_model()


# --- LLM Setup ---
provider = st.sidebar.selectbox("Modelo:", ["gemini", "groq"])
llm = get_llm(provider)
embed_model = cached_embed_model()
configure_settings(llm, embed_model)

# --- File Upload ---
upload = st.sidebar.file_uploader("Envie um documento (PDF ou CSV):", type=["pdf", "csv"])

if upload:
    if st.session_state.docs_list != upload:
        with st.spinner("Processando documento..."):
            suffix = ".pdf" if upload.type == "application/pdf" else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.read())
                path = tmp.name

            if upload.name.endswith(".csv"):
                st.session_state.df = load_csv(path)
                tool, _ = create_pandas_tool(st.session_state.df, llm)
                content = extract_text_from_df(st.session_state.df)
            else:
                st.session_state.df = None
                docs, index = load_and_index(path)
                tool = create_query_engine_tool(index, llm)
                content = extract_text_from_docs(docs)

            math_tools = get_math_tools()
            st.session_state.agent = create_agent([tool] + math_tools, llm)
            st.session_state.docs_list = upload
            st.session_state.summary = summarize_doc(llm, content)

        st.toast("Documento enviado com sucesso", icon="✅")

    # --- Layout ---
    col1, col2 = st.columns([2, 2])

    with col1:
        user_query = st.text_input("Digite sua pergunta:")
        translate_option = st.checkbox("Ativar tradução (PT → EN → PT)")
        send = st.button("Enviar", type="primary")

        if send and user_query:
            query = translate(user_query, "pt", "en") if translate_option else user_query
            response, agent_logs = asyncio.run(run_agent(st.session_state.agent, query))
            result = translate(str(response), "en", "pt") if translate_option else str(response)

            st.divider()
            st.markdown("#### Resposta")
            st.markdown(result)
            with st.expander("Etapas do agente (log)"):
                st.code(agent_logs)

    with col2:
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
        with st.expander("💡 Insights rápidos - Resumo do Documento", expanded=True):
            st.write(st.session_state.summary)
else:
    st.info("Por favor, envie um arquivo para continuar.")
