from langchain_community.document_loaders import WebBaseLoader

def run(targets):
    loader = WebBaseLoader(web_paths=targets)
    return loader.load()
