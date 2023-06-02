from google_doc_client import GoogleDocClient
from dotenv import load_dotenv
import os

load_dotenv()

SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")

docs = GoogleDocClient(service_account_key=SERVICE_ACCOUNT_KEY)

types = {}

ids = [
    "1X4aDPRu5p6TkilBcf4OorTe_PdHjhx7w",
    "14RM72brMzxUR0I8BrrRa40RussjjDaiB",
    "1gvu4aNTC5ahP8fhiZzNg8N845C4U2o25",
    "1WEhn-kQQjYxpWD8hC05Nc4VK9oPYROFl",
    "1WEhn-kQQjYxpWD8hC05Nc4VK9oPYROFl",
    "1-D57rVJClUye-tQGinAqhDDHEpbFjabK",
    # "1pcgDw0I3AXF7YDP6dFttETC6OgaGKAnouURdfwSRyAc",
    "1Mb_p8bK9xt3xGcyrwLVVy1pw30DwEa6r",
    "1fI0H8oWw6zZ4fTaknCshnG_BpGytaSIW",
    "1X4aDPRu5p6TkilBcf4OorTe_PdHjhx7w",
    "1uo7GDueegnCgt87iXznSKHT-Yf5toZwUkFY3DaJww5Y",
    "15pN1pgbPGVciQYGNYEyCxUelnl25pZsM",
    "1rbFrMr7vxaObJjHW8MSzfYpI-vfyRY6kEV3QIPFgF10",
    "1glPH1y0jqL2l6KV6i9nowQHFxKfA1s7VxlUREFIQZWA",
    "10FM8naMdJbD3skrBnuSSa1gHuwhUX1uB",
]

for id in ids:
    t = docs.get_content_type_by_file_id(id)
    types[t] =True
    print(f"id:{id}, type:{t}")
print(types)
