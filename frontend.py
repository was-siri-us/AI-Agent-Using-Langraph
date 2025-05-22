import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon=":robot_face:", layout="wide")
st.title("AI Agent Chatbot")
st.write("This is a simple AI agent chatbot that can answer your questions.")



system_prompt = st.text_area(label="System Prompt", placeholder="You are a helpful assistant.", height=70)

API_URL="http://127.0.0.1:9999/chat"
MODEL_NAMES_GROQ = ["llama3-8b-8192","qwen-qwq-32b"]
    
selected_model = st.selectbox("Select a model", MODEL_NAMES_GROQ)
allow_web_search=st.checkbox("Allow Web Search")
user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")



if st.button("Ask Agent!"):
    if user_query.strip():
        import requests

        payload={
            "model_name": selected_model,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        response=requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")