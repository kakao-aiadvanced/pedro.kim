import json
import configparser

from pipelines import (
    prepare_db,
    determine_relevance
)

def main():
    db = prepare_db.run()
    user_query = "agent memory"
    docs = db.similarity_search(user_query)
    for doc in docs:
        print(determine_relevance.run(user_query, doc))

if __name__ == "__main__":
    main()
