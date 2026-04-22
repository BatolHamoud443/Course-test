# import libraries
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from typing import List, Dict, Any

# load environment variables
load_dotenv()

# App configurations
st.set_page_config(
    page_title="Bitte RAG ChatBot", 
    page_icon=":material/chat_bubble:", # speech bubble icon 
    layout="centered")

# Add a title to the app
st.title("🤖 Bitte RAG ChatBot") # include a bot emoji

# Add a description to the app
st.markdown("**Your intelligent assistant powered by GPT-5 and RAG technology**")
st.divider()

# add a collapsible section
with st.expander("ℹ️ About this webapp", expanded = False):
    st.markdown(
        """
**Bitte RAG Chatbot**

- **Model:** `gpt-5` via OpenAI Responses API

- **RAG:** File Search tool using your pre-built Vector Store

- **Features:** multi-turn chat, image inputs, clear conversation

- **Secrets:** reads `OPENAI_API_KEY` and `VECTOR_STORE_ID` from Streamlit secrets or environment variables

**How it works**

Your message and (optional) images go to the Responses API along with a system resp.
The File Search tool retrieves relevant passages from your vector store to ground the answer.
        """
    )

# Retrieve the credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]

# Set the OpenAI API key in the os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize the OpenAI client
client = OpenAI()

# Warn if OpenAI API key or the vector store id are not set:
if not OPENAI_API_KEY:
    st.warning("OpenAI API key is not set. Please set the OpenAI API key in the environment variables or Streamlit secrets.")
if not VECTOR_STORE_ID:
    st.warning("Vector store ID is not set. Please set the Vector store ID in the environment variables or Streamlit secrets.")


# Configuration of the system prompt:
system_prompt = """
You are a toxic CEO who loves things like pre-revenue or cash burn ratio.
"""

# Store the previous response id
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
    if st.button("Clear Conversation History", use_container_width = True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        # reset the page
        st.rerun()

# Helper functions
def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the input parts array for the OpenAI from text and images.

    Args:
        text: The text to be sent to the OpenAI
        images: The images to be sent to the OpenAI

    Returns:
        A list of input parts compatible with the openAI responses API
    """
    content = []
    if text and text.strip():
        content.append({
            "type": "input_text",
            "text": text.strip()
        })
    for img in images:
        content.append({
            "type": "input_image",
            "image_url": img["data_url"]
        })
    return [{"type": "message", "role": "user", "content": content}] if content else []

# Function to generate a response from the OpenAI responses API
def call_responses_api(parts: List[Dict[str, Any]], previous_response_id: str = None) -> Any:
    """
    Call the OpenAI responses API with the input parts.

    Args:
        parts: The input parts to be sent to the OpenAI
        previous_response_id: The previous response id to be sent to the OpenAI
    """

    tools = [
        {"type": "file_search", 
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": 20}
    ]

    response = client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=system_prompt,
        tools = tools,
        previous_response_id=previous_response_id
    )

    return response

# function to get the text output
def get_text_output(response: Any) -> str:
    """
    Get the text output from the OpenAI responses API.
    """
    return response.output_text

# Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # Extract text content from the message structure
        if isinstance(m["content"], list):
            # Handle structured messages (user input)
            for part in m["content"]:
                if part.get("type") == "message":
                    for content_item in part.get("content", []):
                        if content_item.get("type") == "input_text":
                            st.markdown(content_item["text"])
                        elif content_item.get("type") == "input_image":
                            st.image(content_item['image_url'], width=100)
        else:
            # Handle simple text messages (assistant responses)
            st.markdown(m["content"])

# User interface - upload images
uploaded = st.file_uploader(
    "Upload images", 
    type=["jpg", "jpeg", "png", "webp"], 
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}"
    )
# User interface - chat input
prompt = st.chat_input("Type your message here...")

if prompt is not None:
    # Process only the currently uploaded images into an API-compatible format
    images = []
    if uploaded:
        images = [
            {
                "mime_type": f"image/{f.type.split('/')[-1]}" if f.type else "image/png",
                "data_url": f"data:{f.type};base64,{base64.b64encode(f.read()).decode('utf-8')}"
            }
            for f in uploaded
        ]

    # Build the input parts for the responses API
    parts = build_input_parts(prompt, images)


    # Store the messages
    st.session_state.messages.append({"role": "user", "content": parts})

    # Display the user's message
    with st.chat_message("user"):
        for p in parts:
            if p['type'] == "message":
                for content_item in p.get("content", []):
                    if content_item['type'] == "input_text":
                        st.markdown(content_item['text'])
                    elif content_item['type'] == "input_image":
                        st.image(content_item['image_url'], width=100)
                    else:
                        st.error(f"Unknown content type: {content_item['type']}")

    # Generate the AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_responses_api(parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                # Display the AI's response
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # Retrive the ID if available
                if hasattr(response, "id"):
                    st.session_state.previous_response_id = response.id

            except Exception as e:
                st.error(f"Error generating response: {e}")
