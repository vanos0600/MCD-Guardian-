import streamlit as st
import os
from dotenv import load_dotenv

# Cargar variables
load_dotenv()
secure_key = os.getenv("GOOGLE_API_KEY")

if not secure_key:
    st.error("🚨 FALTA LA CLAVE API EN EL ARCHIVO .ENV")
    st.stop()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN VISUAL (ESTILO MCDONALD'S) ---
st.set_page_config(page_title="McD Guardian", page_icon="🍔", layout="wide")

# CSS CORREGIDO PARA QUE SE VEA EL TEXTO
st.markdown("""
    <style>
    /* 1. Fondo de la aplicación blanco puro */
    .stApp {
        background-color: #ffffff;
    }
    
    /* 2. Forzar que TODO el texto normal sea NEGRO (para arreglar el modo oscuro) */
    p, div, span, li {
        color: #000000 !important;
    }

    /* 3. Títulos en Rojo McDonald's */
    h1, h2, h3 {
        color: #DA291C !important;
    }

    /* 4. Ajuste para que los inputs de texto también se vean bien */
    .stTextInput input {
        color: #000000 !important;
    }
    
    /* 5. Ajuste del contenedor del chat */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    # Logo
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/McDonald%27s_Golden_Arches.svg", width=100)
    st.title("SOP Guardian 🛡️")
    st.markdown("---")
    st.markdown("### 📚 Manuales Activos")
    st.success("✅ Estación de Frituras")
    st.success("✅ Estación de Plancha") 
    st.markdown("---")
    st.info("⚠️ Este sistema es para uso interno. No compartir información confidencial.")

# --- LÓGICA DEL CEREBRO ---
DB_PATH = "./vectorstore"

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_PATH):
        st.error("⚠️ Base de datos no encontrada. Ejecuta ingest.py")
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

def get_rag_chain(vector_db):
    # Usamos el modelo 'flash-latest' que funcionaba bien
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=secure_key)
    
    system_prompt = (
        "Actúa como un Gerente de Entrenadores de McDonald's. "
        "Responde basándote SOLO en el siguiente contexto. "
        "Si no lo sabes, di que no está en el manual. "
        "Sé breve y directo.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

# --- INTERFAZ PRINCIPAL ---
st.title("🍔 Asistente de Operaciones")
st.caption("Pregunta sobre tiempos, temperaturas y procedimientos.")

vector_db = get_vectorstore()

if vector_db:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu SOP Guardian. ¿En qué estación estás trabajando hoy?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: ¿A qué temperatura va la plancha superior?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando manuales..."):
                chain = get_rag_chain(vector_db)
                response = chain.invoke({"input": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})