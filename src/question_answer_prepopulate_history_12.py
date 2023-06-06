from dotenv import load_dotenv
load_dotenv()
import os
# from langchain.document_loaders import GoogleDriveLoader
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.callbacks import get_openai_callback
from langchain.schema import messages_from_dict
from agents import multiagent_expert
import json


# Using the conversational agent that uses retrieval qa as a tool, it takes care of querying the index

def main():

    # os.environ["LANGCHAIN_TRACING"] = "true"


    agent = multiagent_expert(memory_window_size=None, verbose=True)
    # agent = governance_expert()
    # print(agent.agent.llm_chain.prompt)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break

        answer = run_with_token_count(agent, question)

        print(f"Result:\n {json.dumps(answer, indent=4, default=lambda o: o.__dict__)}")
        print(f"Answer:\n {answer['output']}")
        # sprint(f"Sources:\n {sources}")


def run_with_token_count(agent, query):
    chat_history = [
        {
            "data":{ "content": "what is the purpose of the proposal with path polkadot/treasury_proposals/231"},
            "type": "human"
        },
        {
            "data":{ "content": "The purpose of the proposal with path polkadot/treasury_proposals/231 is to provide 6-month maintenance funding for The Kusamarian, a platform that serves token holders by setting a standard for the propagation of vital information and narrative across the Polkadot network. The Kusamarian focuses on software development, community building, and realistic expectations, with an emphasis on fair, truthful, accurate reporting, and journalistic independence. The funding will support the continuation and improvement of their various programs, including Alpha Shots, PNinja on the Road, and Attempts at Governance. SOURCES: [https://docs.google.com/document/d/12b5dPQaPvu-59j2Bf4LoJYRwV_xjIM95t8kCHpMhTPY/edit#bookmark=kix.k2t917l25ohw]"},
            "type": "ai"
        }
    ]
    
    chat_history = messages_from_dict(chat_history)
    # chat_history = "what is the purpose of the proposal with path polkadot/treasury_proposals/231\nThe purpose of the proposal with path polkadot/treasury_proposals/231 is to provide 6-month maintenance funding for The Kusamarian, a platform that serves token holders by setting a standard for the propagation of vital information and narrative across the Polkadot network. The Kusamarian focuses on software development, community building, and realistic expectations, with an emphasis on fair, truthful, accurate reporting, and journalistic independence. The funding will support the continuation and improvement of their various programs, including Alpha Shots, PNinja on the Road, and Attempts at Governance. SOURCES: [https://docs.google.com/document/d/12b5dPQaPvu-59j2Bf4LoJYRwV_xjIM95t8kCHpMhTPY/edit#bookmark=kix.k2t917l25ohw]"
    with get_openai_callback() as cb:
        print(f"Chat history: {chat_history}")
        result = agent({"input": query, "chat_history": chat_history})
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


if __name__ == "__main__":
    main()
