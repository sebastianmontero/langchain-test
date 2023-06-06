from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_util.text_splitter import MarkdownTextSplitterWithContext
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import GovernanceProposal
import re

import pinecone

import json

import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    DB_URL = os.getenv("DB_URL")
    
    engine = create_engine(DB_URL)
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        gp = GovernanceProposal.get_by_id("e0b4893e-7f1f-4395-915a-0c0596bbfe19", session)

    content = gp.content
    print(f"length before removing code sections: {len(content)}")
    content = remove_code_sections(content)
    print(f"length after removing code sections: {len(content)}")
    docs = [
        Document(page_content=content, metadata={"id":gp.governance_proposal_id, "source":gp.governance_proposal_path, "chunk-context": gp.title})
    ]

    text_splitter = MarkdownTextSplitterWithContext(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    # for i, split_doc in enumerate(split_docs):
    #     split_doc.metadata['source'] = f"{split_doc.metadata['url']}-{i}"
    longest_doc = max(split_docs, key=lambda doc: len(doc.page_content))
    print(f"Longest split doc: {longest_doc}")
    print(f"Number of chars in longest doc: {len(longest_doc.page_content)}")
    print(f"Number of docs: {len(split_docs)}")

    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # pinecone.init(
    #     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    #     environment=PINECONE_API_ENV  # next to api key in console
    # )
    # Pinecone.from_documents(documents=split_docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    # print("Stored embeddings.")


def map_to_docs(item) ->Document:
    if item.get("text"):
        metadata = {"type":"docs"}
        context=""
        separator=""
        if item.get("url"):
            metadata["source"] = item["url"]
        if item["metadata"].get("title"):
            title = item["metadata"]["title"]
            to_remove = "· polkadot wiki"
            separator = "\n"
            if title.lower().endswith(to_remove):
                title = title[:-len(to_remove)]
            context += title
        if item["metadata"].get("description"):
            context += separator + item["metadata"]["description"]
        if context:
            metadata["chunk-context"] = context
        return Document(
            page_content=item["text"], metadata=metadata
        )
    return None


def remove_code_sections(markdown_text):
    # Regular expression pattern to match code sections
    code_pattern = r'```.*?```'

    # Remove code sections from the markdown text
    cleaned_text = re.sub(code_pattern, '', markdown_text, flags=re.DOTALL)

    return cleaned_text


if __name__ == "__main__":
    main()



  