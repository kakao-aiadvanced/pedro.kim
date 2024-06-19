from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run(state):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant who answers user queries based "
            "on the contexts given along. A user query and relevant contexts "
            "will be given as follows:\n"
            "\n"
            "<userQuery>user query</userQuery>\n"
            "<contexts>contexts relevant to the user query</contexts>\n"
        ),
        (
            "human",
            "<userQuery>{user_query}</userQuery>\n"
            "<contexts>{contexts}</userQuery>\n"
        )
    ])
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    response = chain.invoke({
        "user_query": state["user_query"],
        "contexts": "\n\n".join(
            doc if isinstance(doc, str) else doc.page_content
            for doc in state["contexts"]
        )
    })
    return {"response": response}
