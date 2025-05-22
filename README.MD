# 🧠 AI Agent Chatbot with LangGraph, GROQ & Streamlit

This project is an interactive AI chatbot built using **FastAPI**, **LangGraph**, and **Streamlit**. It uses **GROQ's large language models** and supports real-time research and web search using tools like **Tavily Search**, **Wikipedia**, and **arXiv**.

## 🌟 Features

- Chat with powerful LLMs like `llama3-8b-8192` and `qwen-qwq-32b` via GROQ
- Plug-and-play search capabilities (Tavily, Wikipedia, ArXiv)
- Easily customizable system prompts for different personas
- Lightweight frontend using Streamlit
- Modular backend built with FastAPI

---

## 📦 Project Structure

```
.
├── agent.py             # Core LangGraph agent logic and tool integrations
├── main.py              # FastAPI app exposing /chat endpoint
├── ui.py                # Streamlit UI for user interaction
├── README.md            # Project documentation
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-agent-chatbot.git
cd ai-agent-chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, here are common dependencies:
```txt
fastapi
uvicorn
streamlit
requests
langchain
langchain-groq
langchain-openai
langgraph
langchain-community
pydantic
```

### 4. Set Environment Variables

Export your API keys in your environment (or use a `.env` file):

```bash
export GROQ_API_KEY="your_groq_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
```

On Windows (CMD):

```cmd
set GROQ_API_KEY=your_groq_api_key
set OPENAI_API_KEY=your_openai_api_key
set TAVILY_API_KEY=your_tavily_api_key
```

---

## 🚀 Run the App

### Step 1: Start the FastAPI Backend

```bash
python main.py
```

This will start the backend server on `http://127.0.0.1:9999`.

### Step 2: Launch the Streamlit Frontend

```bash
streamlit run ui.py
```

This opens up the chatbot UI in your default browser.

---

## 🔄 Example Usage

- Select a model like `qwen-qwq-32b`
- Customize your system prompt (e.g., *"You are a health advisor."*)
- Toggle on “Allow Web Search” to enable Tavily, ArXiv, and Wikipedia