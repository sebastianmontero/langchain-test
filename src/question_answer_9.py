from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.callbacks import get_openai_callback
from agents import governance_expert


# Using the conversational agent that uses retrieval qa as a tool, it takes care of querying the index

def main():
    load_dotenv()

    agent = governance_expert()
    print(agent.agent.llm_chain.prompt)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break

        answer = run_with_token_count(agent, question)

        print(f"Answer:\n {answer}")
        # sprint(f"Sources:\n {sources}")


def run_with_token_count(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


if __name__ == "__main__":
    main()
