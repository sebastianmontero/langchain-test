from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    COMBINE_PROMPT,
    QUESTION_PROMPT,
    EXAMPLE_PROMPT
)

import pinecone

import os


# Using the retrieval qa with sources chain using the most customizable way of creating it, it takes care of querying the index

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

    llm_question_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)
    llm_combine_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
    combine_results_chain = StuffDocumentsChain(
        llm_chain=llm_combine_chain,
        document_prompt=EXAMPLE_PROMPT,
        document_variable_name="summaries",
    )
    combine_document_chain = MapReduceDocumentsChain(
        llm_chain=llm_question_chain,
        combine_document_chain=combine_results_chain,
        document_variable_name="context",
    )
    chain = RetrievalQAWithSourcesChain(combine_documents_chain=combine_document_chain, retriever=vector_store.as_retriever(), return_source_documents=False, verbose=True)
    # print(chain.combine_documents_chain.)
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

  