.

🍔 McD SOP Guardian: AI-Powered Retail Assistant
McD SOP Guardian is a Retrieval-Augmented Generation (RAG) assistant designed to transform how McDonald's operations teams in León, Mexico, interact with their Standard Operating Procedures (SOP).

Instead of flipping through 100+ page physical binders, managers and crew members can now query critical information—like food safety temperatures or equipment troubleshooting—instantly via a mobile-friendly chat interface.

🚀 Key Features
Instant RAG-Based Answers: Direct access to official manuals without hallucinations, powered by ChromaDB.

Retail-Ready UI: Custom-branded interface using McDonald's corporate identity for seamless user adoption.

High Performance: Optimized with Gemini 1.5/2.5 Flash for near-instant response times.

Localized Prototype: Optimized for the specific operational needs of franchises in León, Mexico (Spanish Demo).

🛠️ Technical Stack & Challenges
The Architecture
LLM: Google Gemini 1.5 / 2.5 Flash (via Google AI Studio).

Vector Database: ChromaDB for efficient document indexing and retrieval.

Framework: LangChain for RAG orchestration.

Frontend: Streamlit.

Engineering Challenges Overcome
Environment Stability: Successfully deployed on Python 3.13, managing cutting-edge dependency compatibility.

Infrastructure Patching: Resolved SQLite version conflicts on Linux-based Streamlit Cloud instances to ensure persistent vector storage.

API Resilience: Implemented handling for Error 429 (Rate Limiting), ensuring the application remains robust during high-frequency querying of the Gemini API.

📦 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/mcd-sop-guardian.git
cd mcd-sop-guardian
Install dependencies: (Note: Optimized for Python 3.13)

Bash
pip install -r requirements.txt
Set up Environment Variables: Create a .env file and add your Google AI API Key:

GOOGLE_API_KEY=your_api_key_here

Bash command for running the program: 
streamlit run app.py

📈 Business Impact
Food Safety: Reduces verification time for critical protocols (oil temps, holding times) from minutes to seconds.

Onboarding: Accelerates the training curve for new hires by providing a "digital mentor."

Digital Transformation: Bridges the gap between traditional physical documentation and modern AI-driven operations.

👤 Author
Oskar David Vanegas Juarez 
