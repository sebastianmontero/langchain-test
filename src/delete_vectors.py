from dotenv import load_dotenv
import pinecone
import os
import argparse
from typing import Optional

def delete_vectors(namespace: str, proposal_path: Optional[str]):

    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    # Instantiate a connection to the Pinecone service
    pinecone.init(api_key=PINECONE_API_KEY,
                environment=PINECONE_API_ENV)

    # Connect to an existing index
    index = pinecone.Index(index_name=PINECONE_INDEX_NAME)

    result = None
    if proposal_path:
        print(f"Deleting indexes in namespace: {namespace} and proposal_path: {proposal_path}")
        result = index.delete(
            filter={"governance_proposal_path": {"$eq": proposal_path}},
            namespace=namespace
        ) 
    else:
        print(f"Deleting indexes in namespace: {namespace}")
        result = index.delete(
            delete_all=True,
            namespace=namespace
        )    
    print(f"Result from the deleting operation: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deletes the vectors from a namespace")
    parser.add_argument("namespace", help="The namespace to delete vectors from")
    parser.add_argument('--proposal_path', help='Proposal path argument (optional)')
    args = parser.parse_args()

    delete_vectors(args.namespace, args.proposal_path)