from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from models import GovernanceProposal
from langchain.chains.summarize import load_summarize_chain
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine


import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_URL = os.getenv("DB_URL")
    
    engine = create_engine(DB_URL)
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        gp = GovernanceProposal.get_by_id("e0b4893e-7f1f-4395-915a-0c0596bbfe19", session)


    content = gp.content
    # print(content)
    docs = [Document(page_content=content)]
    text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    print(split_docs[0])
    print(f"Number of chars: {len(split_docs[0].page_content)}")
    print(f"Number of docs: {len(split_docs)}")

    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    # llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    result = chain.run(split_docs)
    print(result)





if __name__ == "__main__":
    main()

  