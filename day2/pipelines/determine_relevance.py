from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run(user_query, document):
    llm = ChatOllama(model="llama3", temperature=0)
    prompt_template = PromptTemplate.from_template(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "\n"
        "You are a helpful AI assistant who compares a user query and a "
        "given context, telling if the context is appropriate for answering "
        "the user query.\n"
        "\n"
        "Example discourses are as follows:\n"
        "\n"
        "<userQuery>John's age</userQuery>\n"
        "<context>John is 42 years old.</context>\n"
        "isContextAppropriate: yes\n"
        "\n"
        "<userQuery>Sarah's age</userQuery>\n"
        "<context>John is 42 years old.</context>\n"
        "isContextAppropriate: no\n"
        "\n"
        "<userQuery>how to build a house</userQuery>\n"
        "<context>Tools for laying out the underpinnings of the structure "
        "can be pretty expensive.  However, they are vital to designing "
        "successful construction projects.</context>\n"
        "isContextAppropriate: yes\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "<userQuery>{user_query}</userQuery>\n"
        "<context>{context}</context>\n"
        "isContextAppropriate: "
    )
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"user_query": user_query, "context": document})
