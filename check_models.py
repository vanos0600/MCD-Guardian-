import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Cargar la clave del archivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: No se encontró la API KEY. Revisa tu archivo .env")
else:
    print(f"🔑 Clave detectada: {api_key[:5]}... (Oculta por seguridad)")
    
    # 2. Configurar Google
    genai.configure(api_key=api_key)

    print("\n📡 Conectando con Google para ver tus modelos disponibles...")
    try:
        found_any = False
        # 3. Listar modelos
        for m in genai.list_models():
            # Solo queremos los que sirven para generar texto (chat)
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ DISPONIBLE: {m.name}")
                found_any = True
        
        if not found_any:
            print("⚠️ No se encontraron modelos compatibles con 'generateContent'.")
            
    except Exception as e:
        print(f"❌ ERROR DE CONEXIÓN: {e}")