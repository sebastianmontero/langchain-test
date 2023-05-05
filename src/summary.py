from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import json

import os


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    # print("api key: ", api_key)
    # loader = UnstructuredURLLoader(urls=["https://centrifuge.hackmd.io/Q4AZOW2WRPq7Q0Ti3ee8Og"])
    # docs = loader.load()
    # print(type(docs[0].page_content))
    # print(docs[0].page_content)
    # print(docs)

    with open("test.json", 'r') as f:
        docj = json.load(f)

    content = docj[0]['div_content']
    # print(content)
    docs = [Document(page_content=content)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    print(split_docs[0])
    print(len(split_docs))

    llm = ChatOpenAI(
      openai_api_key=api_key, 
      model_name="gpt-4"
      )

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

    summary = chain.run(split_docs)

    print("******SUMMARY**************")
    print(summary)

    # loader = GoogleDriveLoader(document_ids=["1WEhn-kQQjYxpWD8hC05Nc4VK9oPYROFl"],
    #                       credentials_path="/home/sebastian/Documents/hashed/gcloud/langchain_oauth_credentials.json",
    #                       token_path="/home/sebastian/Downloads/gd-token.json")
    # docs = loader.load()


if __name__ == "__main__":
    main()

  