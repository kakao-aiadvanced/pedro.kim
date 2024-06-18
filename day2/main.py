import sys
import json
import configparser

from pipelines import (
    prepare_db,
    determine_relevance
)

def main():
    db = prepare_db.run()
    #user_query = "i love kimchi"    # use this query to test if 'hits' become empty
    user_query = "agent memory"
    docs = db.similarity_search(user_query)
    hits = []
    for doc in docs:
        response = determine_relevance.run(user_query, doc)
        if response.strip() == "isContextAppropriate: yes":
            hits.append(doc)
    if not hits:
        print("Sorry, I couldn't find any documents relevant to your query.")
        sys.exit(0)

if __name__ == "__main__":
    main()
