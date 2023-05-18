from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import ApifyDatasetLoader
from langchain_util.text_splitter import RecursiveCharacterTextSplitterWithContext

import pinecone

import json

import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")


    loader = ApifyDatasetLoader(
        dataset_id="0aue91DksrPRRyOVm",
        dataset_mapping_function=map_to_docs)   

    docs = loader.load()


    longest_doc = max(docs, key=lambda doc: len(doc.page_content))
    longest_context = max(docs, key=lambda doc: len(doc.metadata.get("chunk-context")))
    print(f"longest doc: {longest_doc}")
    print(f"longest context: {longest_context}")
    print(f"length of longest doc: {len(longest_doc.page_content)}")
    print(f"length of longest context: {len(longest_context.metadata.get('chunk-context'))}")
    print(f"num docs: {len(docs)}")


    text_splitter = RecursiveCharacterTextSplitterWithContext(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    # for i, split_doc in enumerate(split_docs):
    #     split_doc.metadata['source'] = f"{split_doc.metadata['url']}-{i}"
    longest_doc = max(split_docs, key=lambda doc: len(doc.page_content))
    print(f"Longest split doc: {longest_doc}")
    print(f"Number of chars in longest doc: {len(longest_doc.page_content)}")
    print(f"Number of docs: {len(split_docs)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    Pinecone.from_documents(documents=split_docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    print("Stored embeddings.")


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



if __name__ == "__main__":
    main()

  