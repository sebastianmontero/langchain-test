from dotenv import load_dotenv
from langchain.output_parsers.regex import RegexParser
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAGenerateChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import json

import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    with open("test.json", 'r') as f:
        docj = json.load(f)

    content = docj[0]['div_content']
    # print(content)
    docs = [Document(page_content=content, metadata={"url":"source.com"})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    
    print(split_docs[0])
    print(f"Number of chars: {len(split_docs[0].page_content)}")
    print(f"Number of docs: {len(split_docs)}")

    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    gen_chain = QAGenerateChain.from_llm(llm=llm, verbose=True)
    gen_chain.prompt.output_parser = RegexParser(
        regex=r"QUESTION: (.*?)\n*ANSWER: (.*)", output_keys=["input", "answer"]
    )
    # examples = gen_chain.apply_and_parse([{"doc": content}])
    examples = gen_chain.apply_and_parse([{"doc": doc.page_content} for doc in split_docs])
    with open('qa-eval-dataset.json', 'w') as f:
        json.dump(examples, f)
    
    print("Stored generated QAs.")


if __name__ == "__main__":
    main()

  