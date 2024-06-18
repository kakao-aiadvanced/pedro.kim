import sys
import json
import configparser

from pipelines import (
    prepare_db,
    determine_relevance,
    generate_answer
)

def main():
    print("Preparing vector DB...")
    db = prepare_db.run()
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
    print(generate_answer.run(user_query, hits))

if __name__ == "__main__":
    main()
