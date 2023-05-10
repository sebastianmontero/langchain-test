from dotenv import load_dotenv
from langchain.output_parsers.regex import RegexParser
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from agents import governance_expert
import pandas as pd
import json
import evaluate


import os


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    with open("qa-eval-dataset.json", 'r') as f:
        dataset = json.load(f)

    # dataset = dataset[:1]
    inputs = [{"input": d["input"]} for d in dataset]
    # print(pd.DataFrame(dataset))
    print(json.dumps(dataset, indent=4))
    print(json.dumps(inputs, indent=4))
    agent = governance_expert(memory_window_size=1)

    results = agent.apply(inputs)
    # print(json.dumps(results, indent=4))
    print(results)

    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,
                     model_name="gpt-4", verbose=True)
    eval_chain = QAEvalChain.from_llm(llm=llm)
    eval_results = eval_chain.evaluate(
        dataset, results, question_key="input", answer_key="answer", prediction_key="output")

    print(eval_results)

    for i, dt in enumerate(dataset):
        dt["agent_answer"] = results[i]["output"]
        dt["evaluation"] = eval_results[i]["text"]

    # Results of the evaluation using QAEvalChain
    print("Results of the evaluation using QAEvalChain:\n\n ", json.dumps(dataset, indent=4))

    
    squad_metric = evaluate.load("squad")
    predictions = []
    references = []
    for i, dt in enumerate(dataset):
        predictions.append({
            "id": f"{i}",
            "prediction_text": dt["agent_answer"]
        })
        references.append({
            "id": f"{i}",
            "answers": {
                "answer_start":[0],
                "text":[dt["answer"]]
            }
        })

    results = squad_metric.compute(references=references, predictions=predictions)
    print("Results of the evaluation using Huggingface SQuAD evaluation:\n\n ", json.dumps(results, indent=4))
    # print(squad_metric.description)
    # print(squad_metric.citation)
    # print(squad_metric.features)
    # print(squad_metric.inputs_description)
    # print(f"Number of chars: {len(split_docs[0].page_content)}")
    # print(f"Number of docs: {len(split_docs)}")

    # llm = ChatOpenAI(
    #     temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    # gen_chain = QAGenerateChain.from_llm(llm=llm, verbose=True)
    # gen_chain.prompt.output_parser = RegexParser(
    #     regex=r"QUESTION: (.*?)\n*ANSWER: (.*)", output_keys=["query", "answer"]
    # )
    # # examples = gen_chain.apply_and_parse([{"doc": content}])
    # examples = gen_chain.apply_and_parse([{"doc": doc.page_content} for doc in split_docs])
    # with open('qa-eval-dataset.json', 'w') as f:
    #     json.dump(examples, f)

    # print("Stored generated QAs.")


if __name__ == "__main__":
    main()
