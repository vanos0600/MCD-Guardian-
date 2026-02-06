# 1. PARCHE CRÍTICO PARA STREAMLIT CLOUD (DEBE IR EN LA LÍNEA 1)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Esto permite que el código siga funcionando en local (Windows)
    pass

import streamlit as st
import os
from dotenv import load_dotenv

# 2. CONFIGURACIÓN DE LLM Y SEGURIDAD
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno (.env en local o Secrets en la nube)
load_dotenv()
secure_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="McD Guardian", page_icon="🍔", layout="wide")

# Estilo CSS para forzar colores corporativos y legibilidad
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
    st.error("🚨 Error de Configuración: No se encontró la GOOGLE_API_KEY.")
    st.stop()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/McDonald%27s_Golden_Arches.svg", width=100)
    st.title("SOP Guardian 🛡️")
    st.markdown("---")
    st.markdown("### 📚 Manuales Activos")
    st.success("✅ Estación de Frituras")
    st.success("✅ Estación de Plancha") 
    st.success("✅ Protocolos de Gerencia")
    st.markdown("---")
    st.info("⚠️ Este sistema es para uso interno de entrenamiento.")

# --- LÓGICA DEL MOTOR RAG ---
DB_PATH = "./vectorstore"

@st.cache_resource
def get_vectorstore():
    # Verificamos si la base de datos existe
    if not os.path.exists(DB_PATH):
        st.error("⚠️ Base de datos 'vectorstore' no encontrada en el repositorio.")
        return None
    
    # Modelo de embeddings (conversión de texto a números)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embedding_model
    )

def get_rag_chain(vector_db):
    # Usamos Gemini 1.5 Flash (más rápido y económico)
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0, 
        google_api_key=secure_key
    )
    
    # Instrucciones del sistema para la IA
    system_prompt = (
        "Actúa como un Gerente de Entrenadores de McDonald's experto. "
        "Responde basándote ÚNICAMENTE en el siguiente contexto extraído de los manuales. "
        "Si la información no está en el contexto, di amablemente que no se encuentra en los manuales oficiales. "
        "Usa un tono profesional, breve y directo.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Crear la cadena de procesamiento
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

# --- INTERFAZ DE CHAT ---
st.title("🍔 Asistente de Operaciones")
st.caption("Consulta procedimientos, temperaturas y protocolos de seguridad.")

# Inicializar la base de datos
vector_db = get_vectorstore()

if vector_db:
    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu SOP Guardian. ¿En qué puedo ayudarte hoy?"}
        ]

    # Mostrar mensajes previos
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada de usuario
    if prompt := st.chat_input("Ej: ¿Cómo se prepara un Big Mac?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Consultando manuales oficiales..."):
                try:
                    chain = get_rag_chain(vector_db)
                    response = chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error al conectar con la IA: {e}")