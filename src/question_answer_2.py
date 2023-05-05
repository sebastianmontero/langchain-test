from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain

import pinecone

import os


# Using the retrieval qa with sources chain, it takes care of querying the index

def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )    
    vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
    chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=vector_store.as_retriever(), return_source_documents=True)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break
        
        answer = chain(question)

        print(f"Answer:\n {answer}")
        # sprint(f"Sources:\n {sources}")




if __name__ == "__main__":
    main()

  