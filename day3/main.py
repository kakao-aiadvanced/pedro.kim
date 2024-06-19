import json

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from pipelines.infoquery import graph

def fetch_documents(targets):
    loader = WebBaseLoader(web_paths=targets)
    return loader.load()

def split_documents(targets):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=True,
    )
    return splitter.split_documents(targets)

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
        docs = fetch_documents(documents_to_fetch)
        splitted = split_documents(docs)
        db.add_documents(splitted)
        indexed_documents += documents_to_fetch
        with open("indexed_documents.json", 'w') as f:
            json.dump(indexed_documents, f)
    return db

def main():
    vector_db = prepare_db()
    workflow = graph.build_graph()
    inputs = {"vector_db": vector_db}
    while True:
        try:
            user_query = input("Ask something: ")
            inputs["user_query"] = user_query
        except KeyboardInterrupt:
            break
        for output in workflow.stream(inputs):
            pass

if __name__ == "__main__":
    main()
