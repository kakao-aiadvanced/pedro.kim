import sys
import json
import configparser

from pipelines import (
    prepare_db,
    determine_relevance,
    generate_answer,
    check_hallucination
)

def query(db, retry_count=3):
    #user_query = "i love kimchi"    # use this query to test if 'hits' become empty
    user_query = "agent memory"

    print("Retrieving documents from vector DB...")
    docs = db.similarity_search(user_query)
    hits = []

    print("Checking if the retrieved documents are appropriate...")
    for doc in docs:
        response = determine_relevance.run(user_query, doc)
        if response.strip() == "isContextAppropriate: yes":
            hits.append(doc)
    if not hits:
        print("Sorry, I couldn't find any documents relevant to your query.")
        sys.exit(0)

    print("Generating answer...")
    response = generate_answer.run(user_query, hits)

    print("Checking for hallucination...")
    hallucination_check = json.loads(check_hallucination.run(user_query, hits, response))
    if hallucination_check.get("hallucination") == "yes":
        if retry_count <= 0:
            raise RuntimeError("query failed and no retries left")
        print("The language model detected hallucination, retrying... (retries left: {})".format(retry_count))
        return query(db, retry_count-1)

    return response

def main():
    print("Preparing vector DB...")
    db = prepare_db.run()
    response = query(db)
    print(response)

if __name__ == "__main__":
    main()
