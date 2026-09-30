# MedSync AI Companion

## Overview
MedSync is a Patient Record Simplifier and Health Companion Agent designed to mitigate clinical information overload. Upon receiving a medical document, the agent immediately extracts key telemetry, rewrites medical terminology into plain language, and flags out-of-range values before the user asks a single question. It prioritizes clinical safety by relying on deterministic Python logic for all medical flagging and routing, ensuring the LLM is strictly confined to natural language processing and document-grounded retrieval.

## Core Features
* **Automated Document Ingestion:** Processes PDF medical records instantly using PyPDFLoader and recursive character text splitting to keep clinical values attached to their reference ranges.
* **Deterministic Flagging System:** Evaluates extracted vitals against standard medical reference ranges using strict, rule-based Python logic to prevent LLM hallucinations.
* **Specialist Routing:** Automatically maps flagged abnormal findings to the appropriate medical department using an internal deterministic lookup table.
* **Context-Aware Retrieval Q&A:** Utilizes a FAISS vector store and HuggingFace embeddings to answer user follow-up questions strictly based on the provided personal health record.
* **Production-Grade UI:** A heavily customized Streamlit frontend featuring advanced CSS injection for a responsive, dark-mode workspace, floating chat docks, and fixed telemetry dashboards.

## Technology Stack
* **Frontend:** Streamlit, Custom CSS Injection, Google Sans Typography
* **Backend Framework:** Python, LangChain
* **Vector Store & Embeddings:** FAISS, HuggingFace (BAAI/bge-small-en-v1.5)
* **LLM Engine:** Local/Cloud LLM integrations (e.g., gpt-oss-20b, ChatOllama)

## Installation and Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/codeclogic/medsync.git](https://github.com/codeclogic/medsync.git)
   ```
2. Navigate to the project directory:
   ```bash
   cd medsync
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit server locally:
   ```bash
   streamlit run healthcare_agentic_ai.py --server.port 8501
   ```
5. To expose the application for external testing or presentations, establish a secure tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```

## Disclaimer
This AI is a health companion agent designed for informational purposes and hackathon demonstration only. It operates on synthetic or de-identified data. It does not provide medical diagnoses. Always consult a licensed healthcare professional regarding real medical documents or conditions.
