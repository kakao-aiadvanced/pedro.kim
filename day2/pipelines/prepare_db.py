import json

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from pipelines import (
    fetch_documents,
    split_documents
)

def run():
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
