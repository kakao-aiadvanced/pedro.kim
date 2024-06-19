def run(state):
    docs = state["vector_db"].similarity_search(state["user_query"])
    return {"contexts": docs}
