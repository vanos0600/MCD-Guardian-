import streamlit as st
import os
from dotenv import load_dotenv

# 1. PARCHE DE COMPATIBILIDAD (Importante para Streamlit Cloud)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# 2. IMPORTACIONES DE IA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 3. SEGURIDAD Y LLAVES
load_dotenv()
secure_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

st.set_page_config(page_title="McD Guardian", page_icon="🍔", layout="wide")

if not secure_key:
    st.error("🚨 Error: Configura la GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

# --- ESTILO VISUAL ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    p, div, span, li { color: #000000 !important; }
    h1, h2, h3 { color: #DA291C !important; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/McDonald%27s_Golden_Arches.svg", width=100)
    st.title("SOP Guardian 🛡️")
    st.success("✅ Sistema Conectado")

# --- LÓGICA RAG ---
@st.cache_resource
def load_system():
    # Cargar base de datos
    db_path = "./vectorstore"
    if not os.path.exists(db_path):
        st.error("⚠️ No se encontró la carpeta 'vectorstore'.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Configurar IA
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # <--- USA ESTE NOMBRE EXACTO (Sin -latest)
        temperature=0, 
        google_api_key=secure_key)
    
    # Prompt optimizado para Python 3.13
    prompt = ChatPromptTemplate.from_template("""
    Eres un experto Gerente de McDonald's. Responde usando SOLO este manual:
    {context}
    
    Pregunta: {input}
    """)
    
    # Crear cadenas
    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), combine_chain)

# --- INTERFAZ ---
st.title("🍔 Asistente de Operaciones")

rag_system = load_system()

if rag_system:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu SOP Guardian. ¿Qué duda tienes sobre el manual?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_input := st.chat_input("Escribe tu duda aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Consultando manual..."):
                try:
                    # Usamos el sistema cargado
                    response = rag_system.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error técnico: {str(e)}")