from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from chains import SentimentChain
from  langchain.schema import Document

import os


# Using the conversational agent that uses retrieval qa as a tool, it takes care of querying the index

def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # llm = ChatOpenAI(
    #     temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = SentimentChain.from_llm(llm=llm)
    # docs=[Document(page_content="Amazing community power! Look forward to seeing ecosystem projects working together.")]
    docs=[Document(page_content="Letting the proposer know that if not contextual information on this bounty is available, there is a high chance the bounty will be rejected and the bond slashed.")]
    # docs=[Document(page_content="You should be able to follow any calls for submissions, process rules and prizes here: https://pioneersprize.polkadot.network.")]
    # docs=[Document(page_content="five plus five is 10")]

    result  = chain({'input_documents': docs})
    # result  = chain({'input_documents': docs})
    print(result)



if __name__ == "__main__":
    main()
