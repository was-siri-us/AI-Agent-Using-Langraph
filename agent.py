import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




# openai_llm = ChatOpenAI(model="gpt-4o-min",api_key=OPENAI_API_KEY)
# groq_llm = ChatGroq(model="qwen-qwq-32b")



system_prompt = "Act as an AI Chat bot who is smart and friendly"


def get_response_from_agent(llm_id,query,system_prompt,allow_search=False):
    llm=ChatGroq(model=llm_id)
    search_tool  = TavilySearchResults(max_results=2)
    
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv,description="Query arxiv papers")
    
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=5) 
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki,description="Query Wikipedia")
    
    
    tools = [arxiv, wiki, search_tool] if allow_search else []
    
    agent = create_react_agent(
        model =llm,
        tools=tools,
        state_modifier=system_prompt,
    )
    state = {"messages":query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]









