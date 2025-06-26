import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables as early as possible
load_dotenv()

# Get Google API key from environment
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Show title and description.
st.title("💬 Gemini Chatbot")
st.write(
    "This is a simple chatbot that uses Google's Gemini model to generate responses. "
    "The app automatically uses the Google API key from your .env file."
)

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
        # Start with an empty chat
        st.session_state.chat = model.start_chat(history=[])

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

        # Get response from Gemini
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
        
        # Store the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response.text})
