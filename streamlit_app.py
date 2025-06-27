import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import PyPDF2
import docx
from PIL import Image
import io
import tempfile
import base64

# Load environment variables as early as possible
load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from TXT/RTF
def extract_text_from_text(file):
    return file.getvalue().decode("utf-8")

# Function to get image description or base64
def process_image(file):
    # For RAG purposes, we'll just use the image data directly
    # Returning both the PIL image for display and base64 for the model
    img = Image.open(file)
    buffered = io.BytesIO()
    img.save(buffered, format=img.format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img, f"data:image/{img.format.lower()};base64,{img_str}"

# Function to process any uploaded file
def process_uploaded_file(file):
    # Create a file info dictionary
    file_info = {
        "name": file.name,
        "type": file.type,
        "content": None,
        "display_content": None,
        "is_image": False
    }
    
    # Process based on file type
    if file.type == "application/pdf":
        file_info["content"] = extract_text_from_pdf(file)
        file_info["display_content"] = f"PDF: {file.name} ({len(file_info['content'])} chars)"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_info["content"] = extract_text_from_docx(file)
        file_info["display_content"] = f"DOCX: {file.name} ({len(file_info['content'])} chars)"
    elif file.type == "text/plain" or file.type == "text/rtf" or file.type == "application/rtf":
        file_info["content"] = extract_text_from_text(file)
        file_info["display_content"] = f"Text: {file.name} ({len(file_info['content'])} chars)"
    elif file.type.startswith("image/"):
        img, img_base64 = process_image(file)
        file_info["content"] = img_base64
        file_info["display_content"] = f"Image: {file.name}"
        file_info["is_image"] = True
        file_info["image"] = img
    else:
        file_info["display_content"] = f"Unsupported: {file.name}"
        
    return file_info

# Get Google API key from environment
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Show title and description.
st.title("💬 Gemini Chatbot")
st.write(
    "This is a simple chatbot that uses Google's Gemini model to generate responses. "
    "The app automatically uses the Google API key from your .env file."
)

# Create sidebar with additional instructions input
with st.sidebar:  
    st.header("Bot Settings")
    st.markdown("---")
    
     # Add a button to restart chat with current knowledge base
    if st.button("Restart Chat with Current Settings"):
        st.session_state.messages = []
        st.session_state.pop("chat", None)
        st.rerun()
        
    st.header("Instructions")    
    st.write("Add custom instructions for the model to follow during your conversation.")
    
    # Text area for custom instructions
    if "custom_instructions" not in st.session_state:
        st.session_state.custom_instructions = ""
    
    custom_instructions = st.text_area(
        "Custom Instructions:", 
        value=st.session_state.custom_instructions,
        height=150,
        help="These instructions will be applied to all your conversations with the model."
    )
    
    # Update session state when instructions change
    if custom_instructions != st.session_state.custom_instructions:
        st.session_state.custom_instructions = custom_instructions
        # Reset chat when instructions change
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.session_state.pop("chat", None)
        st.rerun()
    
    st.markdown("---")
    
    # Document upload section
    st.header("Knowledge Base")
    st.write("Upload documents to provide context for the model.")
    
    # Initialize knowledge base in session state if not exist
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "docx", "txt", "rtf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if file is already in knowledge base
            file_exists = any(item["name"] == uploaded_file.name for item in st.session_state.knowledge_base)
            
            if not file_exists:
                # Process the file
                file_info = process_uploaded_file(uploaded_file)
                if file_info["content"]:
                    st.session_state.knowledge_base.append(file_info)
                    st.success(f"Added {uploaded_file.name} to knowledge base")
    
    # Display current knowledge base files
    if st.session_state.knowledge_base:
        st.subheader("Current Knowledge Base")
        for idx, item in enumerate(st.session_state.knowledge_base):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(item["display_content"])
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.knowledge_base.pop(idx)
                    st.rerun()
            
            # Display image preview for images
            if item["is_image"]:
                st.image(item["image"], width=100)
    
    # Clear knowledge base button
    if st.session_state.knowledge_base:
        if st.button("Clear All Documents"):
            st.session_state.knowledge_base = []
            st.rerun()

if not google_api_key:
    st.error("Google API key not found in .env file. Please add GOOGLE_API_KEY to your .env.", icon="🗝️")
else:
    # Configure the Gemini API
    genai.configure(api_key=google_api_key)
    
    # Set up the model
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    
    # Create a session state variable to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chat if it doesn't exist
    if "chat" not in st.session_state:
        # Initialize chat with custom instructions if present
        initial_history = []
        if st.session_state.custom_instructions:
            instructions_text = f"Please follow these instructions for our conversation: {st.session_state.custom_instructions}"
            # Add knowledge base context if available
            if st.session_state.knowledge_base:
                instructions_text += "\n\nI'm also providing you with the following documents as reference. Please use this information to inform your responses:\n\n"
                for item in st.session_state.knowledge_base:
                    if not item["is_image"]:
                        instructions_text += f"--- Document: {item['name']} ---\n{item['content']}\n\n"
            
            initial_history = [
                {"role": "user", "parts": [instructions_text]},
                {"role": "model", "parts": ["I'll follow those instructions and use the provided reference documents throughout our conversation."]}
            ]
        
        # Start chat with instructions if provided
        st.session_state.chat = model.start_chat(history=initial_history)

    # Display the existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field
    if prompt := st.chat_input("What is up?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare message with text and images
        message_parts = []
        
        # Add any images from the knowledge base for this specific message
        image_parts = []
        for item in st.session_state.knowledge_base:
            if item["is_image"]:
                image_parts.append(item["content"])
        
        # First add text prompt
        message_parts.append(prompt)
        
        # Then add images if any
        for img_data in image_parts:
            message_parts.append(img_data)

        # Get response from Gemini
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if len(message_parts) > 1:
                    # With images
                    response = st.session_state.chat.send_message(message_parts)
                else:
                    # Text only
                    response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
        
        # Store the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response.text})
