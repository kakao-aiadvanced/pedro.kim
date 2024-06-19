from typing import List, Any
from typing_extensions import TypedDict

class InfoQueryState(TypedDict):
    user_query: str
    contexts: list
    response: str
    hallucination_check: dict
    vector_db: Any    # TODO: better typing

