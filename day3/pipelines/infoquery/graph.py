from langgraph.graph import StateGraph, END

from .state import InfoQueryState
from .nodes import (
    retrieve_documents,
    determine_relevance,
    generate_answer,
    check_hallucination,
)

def print_answer(state):
    print(state["response"])
    return state

def route_relevance(state):
    if state["contexts"]:
        return "relevant"
    else:
        print("Sorry, I don't have enough knowledge to answer that question.")
    return END

def route_hallucination(state):
    if state["hallucination_check"].get("hallucination") == "yes":
        return "inappropriate"
    return "appropriate"

def build_graph():
    """Build a graph for information query system.

    Query
    -> Retrieve Documents
    -> Relevance Checker
       if yes -> Generate Answer
                 -> Check Hallucination
                    if ok -> Print to User
                    if ng -> Go to Generate Answer
       if no  -> Search Tavily and go to Relevance Checker
    """

    graph_builder = StateGraph(InfoQueryState)
    graph_builder.add_node("retrieve_documents", retrieve_documents.run)
    graph_builder.add_node("determine_relevance", determine_relevance.run)
    graph_builder.add_node("generate_answer", generate_answer.run)
    graph_builder.add_node("check_hallucination", check_hallucination.run)
    graph_builder.add_node("print_answer", print_answer)

    graph_builder.add_edge("retrieve_documents", "determine_relevance")
    graph_builder.add_conditional_edges(
        "determine_relevance",
        route_relevance,
        {
            "relevant": "generate_answer",
            "__end__": "__end__",
            #"irrelevant": "search_tavily"
        }
    )
    graph_builder.add_edge("generate_answer", "check_hallucination")
    graph_builder.add_conditional_edges(
        "check_hallucination",
        route_hallucination,
        {
            "appropriate": "generate_answer",
            "inappropriate": "print_answer"
        }
    )

    graph_builder.set_entry_point("retrieve_documents")
    graph_builder.set_finish_point("print_answer")
    graph = graph_builder.compile()
    return graph

