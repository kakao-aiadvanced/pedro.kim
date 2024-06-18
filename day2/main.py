from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def retrieve_documents(targets):
    loader = WebBaseLoader(web_paths=targets)
    return loader.load()

def get_embeddings(strings):
    openai = OpenAIEmbeddings(model="text-embedding-3-small")
    return openai.embed_documents(strings)

def main():
    documents_to_get = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    docs = retrieve_documents(documents_to_get)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=True,
    )
    splitted = [splitter.split_documents([doc]) for doc in docs]

    #embeddings = [
    #    get_embeddings([x.page_content for x in s]) for s in splitted
    #]

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(splitted[0], embedding_function)
