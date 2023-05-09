from langchain.output_parsers.regex import RegexParser


def main():
    output_parser = RegexParser(
        regex=r"QUESTION: (.*?)\n*ANSWER: (.*)", output_keys=["query", "answer"]
    )
    result = output_parser.parse("""QUESTION: What are some use cases for slow swaps of large volumes in the context of trades proposed through governance?

ANSWER: Some use cases include 1) paying out treasury grants, bounties, or salaries in less volatile currencies (e.g., stablecoins), 2) enabling parachains to build a DOT reserve for acquiring a parachain lease, paying XCM fees, or increasing availability cores during times of high demand, and 3) governance deciding to invest part of the treasury into a token to diversify their treasury.""")

    print("result:", result)


if __name__ == "__main__":
    main()
