import json
import configparser

from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pipelines import (
    fetch_documents,
    split_documents
)

def prepare_db():
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    documents_for_embedding = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    try:
        with open("indexed_documents.json", 'r') as f:
            indexed_documents = json.load(f)
    except IOError:
        indexed_documents = []
    documents_to_fetch = [
        url for url in documents_for_embedding
        if url not in indexed_documents
    ]
    if documents_to_fetch:
        docs = fetch_documents.run(documents_to_fetch)
        splitted = split_documents.run(docs)
        db.add_documents(splitted)
        indexed_documents += documents_to_fetch
        with open("indexed_documents.json", 'w') as f:
            json.dump(indexed_documents, f)
    return db

def determine_relevance(user_query, document):
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

def main():
    db = prepare_db()
    user_query = "agent memory"
    docs = db.similarity_search(user_query)
    for doc in docs:
        print(determine_relevance(user_query, doc))

if __name__ == "__main__":
    main()
