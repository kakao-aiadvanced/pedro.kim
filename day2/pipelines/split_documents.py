from langchain_text_splitters import RecursiveCharacterTextSplitter

def run(targets):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=True,
    )
    return splitter.split_documents(targets)
