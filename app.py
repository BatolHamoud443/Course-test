# Import libraries
import base64
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)
ENV_FILE_EXISTS = ENV_PATH.exists()
ENV_FILE_SIZE = ENV_PATH.stat().st_size if ENV_FILE_EXISTS else 0


def get_secret(name: str) -> str:
    """Read Streamlit secrets only when a secrets file exists."""
    project_secret_path = Path(".streamlit/secrets.toml")
    user_secret_path = Path.home() / ".streamlit/secrets.toml"

    if not project_secret_path.exists() and not user_secret_path.exists():
        return ""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or get_secret("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or get_secret("VECTOR_STORE_ID")

# App configuration
st.set_page_config(
    page_title="Bitte RAG ChatBot",
    page_icon=":material/chat_bubble:",
    layout="centered",
)

st.title("🤖 Bitte RAG ChatBot")
st.markdown("**Your intelligent Bitte RAG Assistant**")
st.divider()

with st.expander("📚 App Instructions", expanded=False):
    st.markdown(
        """
        - Ask questions about the Bitte dataset in natural language.
        - Attach images to include visual context in your prompt.
        - The assistant uses RAG over your configured vector store.
        """
    )

if not OPENAI_API_KEY:
    st.warning(
        "OpenAI API key not found. Set OPENAI_API_KEY in your .env file "
        "or in .streamlit/secrets.toml."
    )
if not VECTOR_STORE_ID:
    st.warning(
        "Vector store ID not found. Set VECTOR_STORE_ID in your .env file "
        "or in .streamlit/secrets.toml."
    )
if ENV_FILE_EXISTS and ENV_FILE_SIZE == 0:
    st.error(
        f"Detected empty .env file at {ENV_PATH}. Save your keys to this file "
        "and restart the app."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

with st.sidebar:
    st.header("User Controls")
    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.rerun()


def build_input_parts(text: str, images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build OpenAI input payload from text + images."""
    content: List[Dict[str, Any]] = []
    if text.strip():
        content.append({"type": "input_text", "text": text.strip()})
    for img in images:
        content.append({"type": "input_image", "image_url": img["data_url"]})
    return [{"type": "message", "role": "user", "content": content}] if content else []


def call_responses_api(parts: List[Dict[str, Any]], previous_response_id: str = None) -> Any:
    """Call OpenAI Responses API."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    tools: List[Dict[str, Any]] = []
    if VECTOR_STORE_ID:
        tools = [
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 20,
            }
        ]
    return client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=(
            "You are a helpful assistant that answers questions about the Bitte dataset."
        ),
        tools=tools,
        previous_response_id=previous_response_id,
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
prompt = st.chat_input("Type your message here...")

if prompt:
    images = []
    for file in uploaded_files or []:
        mime_type = file.type or "image/png"
        data_url = f"data:{mime_type};base64,{base64.b64encode(file.read()).decode('utf-8')}"
        images.append({"mime_type": mime_type, "data_url": data_url})

    parts = build_input_parts(prompt, images)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        for img in images:
            st.image(img["data_url"], width=200)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not OPENAI_API_KEY:
                st.error("Cannot call OpenAI API: missing OPENAI_API_KEY.")
            else:
                try:
                    response = call_responses_api(
                        parts, st.session_state.previous_response_id
                    )
                    output_text = response.output_text or "No response text returned."
                    st.markdown(output_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output_text}
                    )
                    if hasattr(response, "id"):
                        st.session_state.previous_response_id = response.id
                except Exception as exc:
                    st.error(f"Error generating response: {exc}")
    
