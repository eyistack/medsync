import os
import re
import tempfile
from typing import List
from dotenv import load_dotenv

# CRITICAL RULE 2: Ensure load_dotenv remains at the top
load_dotenv()

import streamlit as st
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(layout="wide", page_title="MedSync AI Companion")

def load_medical_pdf(pdf_path: str):
    """Load a medical PDF, chunk it, embed it, and return a FAISS retriever."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="all-minilm")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


@tool
def flagging_tool(glucose: float, heart_rate: float, systolic_bp: float) -> str:
    """Check numerical vitals (Glucose, Heart Rate, Systolic BP) and flag out-of-range results. Pass 0.0 if a vital is missing."""
    flags = []

    if glucose > 99.0:
        flags.append(f"High Glucose detected: {glucose}")

    if heart_rate > 100.0:
        flags.append(f"High Heart Rate detected: {heart_rate}")
        
    if systolic_bp > 120.0:
        flags.append(f"High Blood Pressure detected: {systolic_bp}")

    if not flags:
        return "No out-of-range values were found for Glucose, Heart Rate, or Systolic BP."

    return "Flagged findings:\n" + "\n".join(flags)


@tool
def specialist_finder(flagged_issue: str) -> str:
    """Map flagged issue names to relevant specialists using deterministic lookup."""
    specialist_map = {
        "high glucose": "Endocrinologist",
        "high heart rate": "Cardiologist",
        "high blood pressure": "Cardiologist"
    }

    normalized_issue = flagged_issue.strip().lower()
    recommendations = set()

    for issue, specialist in specialist_map.items():
        if issue in normalized_issue:
            recommendations.add(specialist)

    if recommendations:
        return f"Recommended specialist: {', '.join(recommendations)}"
    
    return "Recommended specialist: General Practitioner"

SYSTEM_PROMPT = """
You are a helpful medical AI assistant. Answer the user's specific follow-up questions directly, conversationally, and accurately based on the retrieved context. Do NOT output structured templates like 'Flagged Findings' or 'Specialist Recommendation' for general follow-up questions.
Never provide a medical diagnosis. Keep responses factual and grounded in the provided document text.
""".strip()

def build_agent():
    # CRITICAL RULE 1: Preserve model string exactly
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    tools = [flagging_tool, specialist_finder]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def _agent_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response generated."

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block.get("text", ""))
            else:
                text_parts.append(str(block))
        return "\n".join([part for part in text_parts if part]).strip()

    return str(content)

def _extract_num(label: str, text: str) -> float:
    pattern = rf"{label}\s*(?:Level|Rate)?\s*[:=-]?\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else 0.0


def _parse_telemetry(context_text: str, agent_text: str = "") -> dict:
    """Dynamically parses the AI output block and context to populate analytics cards."""
    telemetry = {
        "Glucose": 0.0,
        "Heart Rate": 0.0,
        "Body Temp": 0.0,
        "Blood Pressure": "--",
        "Specialist": "Pending Review",
        "Medications": "Awaiting Data"
    }
    
    combined_text = agent_text + "\n" + context_text

    meds_match = re.search(r"Suggested Medicines\s*[:=-]\s*([^\n]+)", agent_text, flags=re.IGNORECASE)
    if meds_match:
        val = meds_match.group(1).strip()
        telemetry["Medications"] = val if val.lower() not in ["none", "not recorded"] else "None Detected"

    spec_match = re.search(r"Recommended Specialist\s*[:=-]\s*([^\n]+)", agent_text, flags=re.IGNORECASE)
    if spec_match:
        val = spec_match.group(1).strip()
        telemetry["Specialist"] = val if val.lower() not in ["none", "not recorded"] else "General Practitioner"

    bp_match = re.search(r"Blood Pressure\s*[:=-]\s*([\d]{2,3}/[\d]{2,3})", combined_text, flags=re.IGNORECASE)
    if bp_match:
        telemetry["Blood Pressure"] = bp_match.group(1).strip()

    telemetry["Glucose"] = _extract_num("Glucose", combined_text)
    telemetry["Heart Rate"] = _extract_num("Heart Rate", combined_text)
    telemetry["Body Temp"] = _extract_num(r"(?:Body\s*)?Temp(?:erature)?", combined_text)
    
    if telemetry["Specialist"] == "Pending Review":
        if "cardiologist" in combined_text.lower(): telemetry["Specialist"] = "Cardiologist"
        elif "endocrinologist" in combined_text.lower(): telemetry["Specialist"] = "Endocrinologist"
        
    if telemetry["Medications"] == "Awaiting Data":
        meds = []
        if "metformin" in combined_text.lower(): meds.append("Metformin")
        if "lisinopril" in combined_text.lower(): meds.append("Lisinopril")
        if "insulin" in combined_text.lower(): meds.append("Insulin")
        telemetry["Medications"] = ", ".join(meds) if meds else "None Detected"

    return telemetry

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a medical PDF to begin automated review."}]

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "Glucose": 0.0, "Heart Rate": 0.0, "Body Temp": 0.0,
        "Blood Pressure": "--", "Specialist": "Pending Review",
        "Medications": "Awaiting Data"
    }

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

st.markdown("""
<style>
/* Hide default Streamlit overlays */
[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Apple + Claude Dark Base Theme */
.stApp {
    background-color: #181715 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Force sidebar to the very front */
[data-testid="stSidebar"] {
    z-index: 999999 !important; 
}

/* Sidebar Dead Space Elimination */
[data-testid="stSidebarHeader"] {
    display: none !important;
    padding: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 1rem !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow-y: hidden !important; /* Kills the unneeded scrollbar */
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important; 
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 0rem !important;
}
[data-testid="stFileUploader"] {
    margin-bottom: 0.5rem !important;
}
hr {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* Telemetry Cards CSS */
.telemetry-card {
    background: #252320 !important;
    border: 1px solid rgba(230, 223, 216, 0.12) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin-bottom: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Recommendation Card CSS */
.recommendation-card {
    background: #252320 !important;
    border: 1px solid rgba(230, 223, 216, 0.12) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin-bottom: 12px !important; /* Forces a gap between stacked cards */
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Main Viewport Tightening */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 120px !important;
}
.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 120px !important;
    max-width: 100% !important;
}

/* Chat Input Bottom Dock Optimization */
[data-testid="stBottom"] {
    background-color: #181715 !important;
    padding-bottom: 24px !important;
    padding-top: 16px !important;
}
[data-testid="stBottom"] > div {
    background-color: transparent !important;
}
.stChatInput {
    background: transparent !important;
}
.stChatInput > div {
    background-color: #252320 !important;
    border: 1px solid rgba(230, 223, 216, 0.15) !important;
    border-radius: 9999px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important; /* Adds a subtle Apple-style float shadow */
}
[data-testid="stChatInput"] textarea {
    color: #faf9f5 !important;
}

/* Chat Message Card styling */
[data-testid="stChatMessage"] {
    background-color: #252320 !important;
    border: 1px solid rgba(230, 223, 216, 0.12) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    color: #faf9f5 !important;
}

/* 1. Typography (Safer targeting) */
html, body, p, h1, h2, h3, span, div {
    font-family: 'Google Sans', sans-serif !important;
}

/* 2. Protect Streamlit internal icons from the font override */
.material-symbols-rounded, 
.material-icons, 
[data-testid="stIconMaterial"],
.stIcon {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-variant-ligatures: normal !important;
}

/* 3. Custom Dark Scrollbars */
::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
::-webkit-scrollbar-track {
    background: transparent !important;
}
::-webkit-scrollbar-thumb {
    background: #332f2b !important;
    border-radius: 10px !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #5db872 !important;
}

/* 3. Card Hover Effects */
.telemetry-card, .recommendation-card {
    transition: border-color 0.2s ease, transform 0.2s ease !important;
}
.telemetry-card:hover, .recommendation-card:hover {
    border-color: #5db872 !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 24px;">
    <div style="flex-grow: 1;">
        <span style="font-family: 'Georgia', serif; font-size: 22px; color: #faf9f5; font-weight: 400;">MedSync AI Companion</span>
        <span style="font-family: 'Inter', sans-serif; color: #a09d96; font-size: 13px; margin-left: 12px;">Clinical Intelligence Workspace</span>
    </div>
    <div style="display: flex; justify-content: flex-end; align-items: center;">
        <div style="background-color: #252320; border: 1px solid rgba(230, 223, 216, 0.12); border-radius: 9999px; padding: 6px 14px; color: #faf9f5; font-size: 11px; font-weight: 500; display: flex; align-items: center; gap: 6px; white-space: nowrap;">
            <span style="height: 6px; width: 6px; background-color: #5db872; border-radius: 50%; display: inline-block;"></span> Online / Verified Session
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

def render_metric_card(label, value, is_rec=False):
    card_class = "recommendation-card" if is_rec else "telemetry-card"
    center_style = " justify-content: center;" if is_rec else ""
    val_style = " margin: auto 0;" if is_rec else ""
    return f"""
    <div class="{card_class}" style="display: flex; flex-direction: column; gap: 6px; height: 100%;{center_style}">
        <div style="font-size: 10px; text-transform: uppercase; color: #a09d96; font-weight: 600; letter-spacing: 0.5px;">{label}</div>
        <div style="font-size: 16px; font-weight: 700; color: #faf9f5;{val_style}">{value}</div>
    </div>
    """

with st.sidebar:
    st.markdown("<h3 style='color: #faf9f5; font-size: 16px;'>Document Ingestion</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Medical Report", type=["pdf"])

    st.divider()
    st.markdown("<div style='margin-bottom: 12px; color: #a09d96; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Clinical Telemetry & Vitals</div>", unsafe_allow_html=True)

    tel = st.session_state.telemetry
    sc1, sc2 = st.columns(2)
    with sc1: st.markdown(render_metric_card("Glucose Level", f"{tel['Glucose']:.0f} mg/dL" if tel['Glucose'] > 0 else "Awaiting Data"), unsafe_allow_html=True)
    with sc2: st.markdown(render_metric_card("Heart Rate", f"{tel['Heart Rate']:.0f} bpm" if tel['Heart Rate'] > 0 else "Awaiting Data"), unsafe_allow_html=True)
    
    sc3, sc4 = st.columns(2)
    with sc3: st.markdown(render_metric_card("Body Temp", f"{tel['Body Temp']:.1f} F" if tel['Body Temp'] > 0 else "Pending"), unsafe_allow_html=True)
    with sc4: st.markdown(render_metric_card("Blood Pressure", tel.get('Blood Pressure', '-- mmHg')), unsafe_allow_html=True)

    st.write("") # Spacer
    st.markdown(render_metric_card("Prescribed / Suggested Medicines", tel.get('Medications', 'None Detected'), is_rec=True), unsafe_allow_html=True)
    st.markdown(render_metric_card("Specialist Referral", tel.get('Specialist', 'Pending Review'), is_rec=True), unsafe_allow_html=True)

if uploaded_file is not None and st.session_state.get("active_file") != uploaded_file.name:
    with st.spinner("Processing PDF and generating clinical summary..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        retriever = load_medical_pdf(tmp_path)
        st.session_state.retriever = retriever
        st.session_state.active_file = uploaded_file.name

        docs = retriever.invoke("Summarize key patient findings including Glucose, Heart Rate, Temperature, and Blood Pressure")
        context = "\n\n".join(doc.page_content for doc in docs)
        st.session_state.latest_context = context

        prompt = (
            "Provide a plain-language rewrite of the uploaded medical record and a list of flagged items. "
            "You must extract numerical vitals and run flagging_tool first, and specialist_finder for each flagged issue before summarizing. "
            "Never provide a medical diagnosis.\n\n"
            "CRITICAL: At the end of your response, you MUST output a telemetry block exactly in this format:\n"
            "Telemetry Data:\n"
            "- Glucose: [Extract value or 'Not Recorded']\n"
            "- Heart Rate: [Extract value or 'Not Recorded']\n"
            "- Body Temperature: [Extract value or 'Not Recorded']\n"
            "- Blood Pressure: [Extract value or 'Not Recorded']\n"
            "- Suggested Medicines: [Extract comma-separated list or 'None']\n"
            "- Recommended Specialist: [Extract specialist name or 'General Practitioner']\n\n"
            f"Retrieved medical text:\n{context}"
        )
        result = st.session_state.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        summary = _agent_text(result)

        st.session_state.telemetry = _parse_telemetry(context, summary)
        st.session_state.messages.append({"role": "assistant", "content": summary})
        st.rerun()

st.caption("⚠️ **Disclaimer:** This AI is a health companion for informational purposes only. It does not provide medical diagnoses. Always consult a licensed healthcare professional.")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Bottom padding

if prompt := st.chat_input("Ask about clinical findings, risks, or medical history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)

    with st.spinner("Analyzing patient data..."):
        retriever = st.session_state.get("retriever")
        if retriever is None:
            answer = "Please upload a medical PDF in the sidebar first so I can analyze the patient record."
            context = ""
        else:
            docs = retriever.invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)
            st.session_state.latest_context = context

            agent_prompt = (
                f"Context from medical record:\n{context}\n\n"
                f"User question: {prompt}"
            )
            result = st.session_state.agent.invoke({"messages": [{"role": "user", "content": agent_prompt}]})
            answer = _agent_text(result)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set GROQ_API_KEY before running this script.")