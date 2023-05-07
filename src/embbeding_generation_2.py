from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pinecone

import json

import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    

    with open("test.json", 'r') as f:
        docj = json.load(f)

    content = docj[0]['div_content']
    # print(content)
    docs = [Document(page_content=content, metadata={"url":"source.com"})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    for i, split_doc in enumerate(split_docs):
        split_doc.metadata['source'] = f"{split_doc.metadata['url']}-{i}"
    print(split_docs[0])
    print(f"Number of chars: {len(split_docs[0].page_content)}")
    print(f"Number of docs: {len(split_docs)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    Pinecone.from_documents(documents=split_docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    print("Stored embeddings.")




if __name__ == "__main__":
    main()

  