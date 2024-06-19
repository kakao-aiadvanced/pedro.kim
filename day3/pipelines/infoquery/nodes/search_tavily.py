import os

from tavily import TavilyClient

def run(state):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(state["user_query"])
    return {
        "contexts": [
            "Title: {title}\n"
            "Content: {content}\n".format(
                title=doc["title"], content=doc["content"]
            ) for doc in response["results"]
        ]
    }
