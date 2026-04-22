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

# App configuration
st.set_page_config(
    page_title="Bitte RAG ChatBot",
    page_icon=":material/chat_bubble:",
    layout="centered",
)

# App title
st.title("🤖 Bitte RAG ChatBot")

# App description
st.markdown("**Your intelligent Bitte RAG Assistant**")
st.divider()

# Collapsible instructions section
with st.expander("📚 App Instructions", expanded=False):
    st.markdown(
        """
        - This app is a RAG chatbot that uses the Bitte dataset to answer questions.
        - You can ask questions about the Bitte dataset.
        - You can also ask questions about the Bitte dataset.
        - You can also ask questions about the Bitte dataset.
        """
    )

# Retru
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

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Warn if the API key or the vector store ID is not set
if not OPENAI_API_KEY:
    st.warning(
        "OpenAI API key not found. Set OPENAI_API_KEY in your .env file or "
        "in .streamlit/secrets.toml."
    )
 
if not VECTOR_STORE_ID:
    st.warning(
        "Vector store ID not found. Set VECTOR_STORE_ID in your .env file or "
        "in .streamlit/secrets.toml."
    )

if ENV_FILE_EXISTS and ENV_FILE_SIZE == 0:
    st.error(
        f"Detected empty .env file at {ENV_PATH}. Save your keys to this file "
        "and restart the app."
    )
 
client = OpenAI()

# configuration of system prompt
system_prompt = """
You are a helpful assistant that can answer questions about the Bitte dataset.
You are using the Bitte dataset to answer questions.
"""


# Initialize the previous response ID
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a sidebar with user controls
with st.sidebar:
    st.header("User Controls")
    st.divider()
    # Clear the conversation history - reset chat history and context
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.rerun()


# Helper function
def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the input array for the OpenAI from text and images
    Arg:
        text: The text to be sent to the OpenAI
        images: The images to be sent to OpenAI

    Returns:
        A list of input parts compatible with the openAI response API
    """
    content = []
    if text and text.strip():
        content.append(
            {
                "type": "input_text",
                "text": text.strip(),
            }
        )
    for img in images:
        content.append(
            {
                "type":"input_image",
                "image_url": img["data_url"]
            }
        )
    return [{"type": "message", "role": "user", "content": content}] if content else []


# Function to generate a response from OpenAI Responses API
def call_responses_api(parts: List[Dict[str, Any]], previous_response_id: str = None) -> Any:
    """
    Call the OpenAI responses API with the input parts.
    Args: 
        parts: The input parts to be sent to the OpenAI
        previous_response_id: The previous response if to be sent to the OpenAI
    """

    tools = [
        {
            "type":"file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 20,
        }
    ]
    response = client.responses.create(
        model = "gpt-5-nano",
        input = parts,
        instructions = system_prompt,
        tools = tools,
        previous_response_id = previous_response_id

    )
    
    return response

# Function to get the text output
def get_text_output(response:Any) -> str:
    return response.output_text

# Render all previous messages
for m in st.session_state.messages:
        with st.chat_message(m["role"]):
        if isinstance(m['content'], list):
            for part in m['content']:
                if p.get("type") == "message":
                    for content_item in p.get("content",[]):
                        if content_item['type'] == "input_text":
                            st.markdown(content_item['text'])
                        elif content_item['type'] == "input_image":
                            st.image(content_item['image_url'], width =100)
                        
                        else:
                            st.error(f"Unkown content type: {content_item['type']}")


# User interface - upload images
uploaded = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key = f"file uploader_{len(st.session_state.messages)}")

prompt = st.chat_input("Type your message here...")

if prompt is not None:
    # Process the images into an API-compatible format

    images = []
    if uploaded = 
        images =[
            {
                "mime_type": f"image/{f.type.split('/')[-1]} " if f.type else 'image/png',
                "data_url": f"data:{f.type};base64,{base64.b64encode(f.read()).decode('utf-8')}"


            }
        for f in uploaded
    ]

    parts = build_input_parts(prompt, images)

    # Store the messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message('user'):
        for p in parts:
            if p["type"] == "message":
                for content_item in p.get("content",[]):
                    if content_item['type'] == "input_text":
                        st.markdown(content_item['text'])
                    elif content_item['type'] == "input_image":
                        st.image(content_item['image_url'], width =100)
                       
                    else:
                        st.error(f"Unkown content type: {content_item['type']}")




    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_responses_api(parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                if hasattr(response, "id"):
                    st.session_state.previous_response_id = response.id

            except Exception as e:
                st.error(f"Error generating response: {e}")
    
