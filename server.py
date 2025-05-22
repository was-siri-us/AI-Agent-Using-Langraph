from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, Request, Response
from agent import get_response_from_agent




class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]
    allow_search: bool
    
    
app = FastAPI(title="LanGraph Agent")

ALLOWED_MODELS = ["llama3-8b-8192","qwen-qwq-32b"]
@app.post("/chat")
def chat_endpoint(request:RequestState):
    """
    API endpoint to interacat with GROQ using LangGraph and search tools.
    """
    
    if request.model_name not in ALLOWED_MODELS:
        return {"error":"Invalid Model Name. Kindly select an allowed model."}
    llm_id = request.model_name
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    query = request.messages
    
    response = get_response_from_agent(llm_id,query,system_prompt,allow_search)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=9999)
    
    
