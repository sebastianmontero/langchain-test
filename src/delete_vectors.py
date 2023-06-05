from dotenv import load_dotenv
import pinecone
import os
import argparse

def delete_vectors(namespace: str):

    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    # Instantiate a connection to the Pinecone service
    pinecone.init(api_key=PINECONE_API_KEY,
                environment=PINECONE_API_ENV)

    # Connect to an existing index
    index = pinecone.Index(index_name=PINECONE_INDEX_NAME)

    index.delete(
        delete_all=True,
        namespace=namespace
    )    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deletes the vectors from a namespace")
    parser.add_argument("namespace", help="The namespace to delete vectors from")
    args = parser.parse_args()

    delete_vectors(args.namespace)