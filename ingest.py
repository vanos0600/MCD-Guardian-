import os
import shutil
# Importaciones actualizadas para las versiones nuevas
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuración de directorios
DATA_PATH = "./data/manual_ficticio.txt"
DB_PATH = "./vectorstore"

def create_vector_db():
    print(f"🔄 Iniciando ingesta de datos desde: {DATA_PATH}")

    # 1. Verificar que el archivo existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: No encuentro el archivo en {DATA_PATH}")
        print("   Asegúrate de haber creado la carpeta 'data' y dentro 'manual_ficticio.txt'")
        return

    # 2. Cargar documento
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"📄 Documento cargado correctamente.")

    # 3. Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Texto dividido en {len(chunks)} fragmentos.")

    # 4. Crear Embeddings (usando la librería nueva)
    print("🧠 Descargando modelo de IA (esto tarda un poco la primera vez)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Guardar en ChromaDB
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Limpiar DB anterior si existe

    print("💾 Guardando en base de datos vectorial...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"✅ ¡ÉXITO! Base de datos creada en: {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()