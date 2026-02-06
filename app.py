# 1. PARCHE CRÍTICO PARA BASES DE DATOS EN LA NUBE (LÍNEA 1)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from dotenv import load_dotenv

# 2. IMPORTACIONES DE INTELIGENCIA ARTIFICIAL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables (.env en local o Secrets en Streamlit Cloud)
load_dotenv()
secure_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="McD Guardian", page_icon="🍔", layout="wide")

# Forzar colores corporativos (Texto negro sobre fondo blanco)
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    p, div, span, li { color: #000000 !important; }
    h1, h2, h3 { color: #DA291C !important; }
    .stTextInput input { color: #000000 !important; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

if not secure_key:
    st.error("🚨 Error: No se encontró la GOOGLE_API_KEY en Secrets o .env")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/McDonald%27s_Golden_Arches.svg", width=100)
    st.title("SOP Guardian 🛡️")
    st.markdown("---")
    st.markdown("### 📚 Manuales Activos")
    st.success("✅ Estación de Frituras")
    st.success("✅ Estación de Plancha") 
    st.success("✅ Protocolos de Gerencia")

# --- LÓGICA RAG ---
DB_PATH = "./vectorstore"

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_PATH):
        st.error("⚠️ La carpeta 'vectorstore' no está en el repositorio de GitHub.")
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

def get_rag_chain(vector_db):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=secure_key)
    system_prompt = (
        "Eres un Gerente de McDonald's experto. Responde basándote SOLO en este contexto:\n"
        "{context}\n\n"
        "Si no está aquí, di que no se encuentra en el manual. Sé breve."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), qa_chain)

# --- INTERFAZ ---
st.title("🍔 Asistente de Operaciones")

vector_db = get_vectorstore()

if vector_db:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu SOP Guardian. ¿Qué duda tienes sobre el manual?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: ¿Temperatura del aceite?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                chain = get_rag_chain(vector_db)
                response = chain.invoke({"input": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})