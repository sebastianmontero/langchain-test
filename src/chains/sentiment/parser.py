from langchain.output_parsers import  StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="sentiment",
                   description="The sentiment of the statement, -1 for Negative, 0 for Neutral, 1 for Positive"),
    ResponseSchema(name="confidence", description="The confidence level, a value between 0 and 10"),
]

PARSER=StructuredOutputParser.from_response_schemas(response_schemas)