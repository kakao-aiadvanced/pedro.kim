import os

from tavily import TavilyClient

def run(state):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(state["user_query"])
    return {
        "contexts": response["results"],
        "searched_from_tavily": True
    }
