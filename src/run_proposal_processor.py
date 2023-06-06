import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from str_filters import StrContainsFilter
from url_filters import DomainUrlFilter, ContentTypeFilter
from google_doc_client import GoogleDocClient
from scrapers import GoogleExternalDocScraper, GoogleNativeDocScraper, GoogleDriveFileScraper, FileUrlScraper, WebPageScraper
from proposal_detector import KeywordProposalDetector
from proposal_processor import ProposalProcessor
import logging
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from document_stores import VectorDocumentStore
from summarizers import Summarizer

load_dotenv()
# logging.basicConfig(level=logging.DEBUG)


def main():
    DB_URL = os.getenv("DB_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    SERVICE_ACCOUNT_KEY = os.getenv("SERVICE_ACCOUNT_KEY")
    PROPOSAL_NAMESPACE = os.getenv("PROPOSAL_NAMESPACE")

    llm = ChatOpenAI(
      openai_api_key=OPENAI_API_KEY, 
      model_name="gpt-4"
      )
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )

    url_filters = [
        DomainUrlFilter(domains_to_filter=[
            "youtube.com",
            "www.youtube.com",
            "polkadot.subscan.io",
            "twitter.com",
            "diamondstandard.co",
            "nuponixlabs.com",
            "www.benzinga.com"
        ]),
        StrContainsFilter(contents_to_filter=[
            "docs.google.com/spreadsheets",
            "polkassembly.io"
        ]),
        ContentTypeFilter(content_types_to_filter=[
            "image"
        ])
    ]
    content_filters = [
        StrContainsFilter(contents_to_filter=[
            "if you own this account, login and tell us more about your proposal"
        ]),
    ]
    google_doc_client = GoogleDocClient(service_account_key=SERVICE_ACCOUNT_KEY)
    scrapers = [
        GoogleNativeDocScraper(
            google_doc_client=google_doc_client
        ),
        GoogleExternalDocScraper(
            google_doc_client=google_doc_client
        ),
        GoogleDriveFileScraper(),
        FileUrlScraper(),
        WebPageScraper()
    ]

    proposal_processor = ProposalProcessor(
        db_url=DB_URL,
        proposal_score_threshold=0.68,
        url_filters=url_filters,
        content_filters=content_filters,
        scrapers=scrapers,
        proposal_detector=KeywordProposalDetector(),
        document_store= VectorDocumentStore(
            embeddings=embeddings,
            vector_store=Pinecone,
            index_name=PINECONE_INDEX_NAME,
            namespace=PROPOSAL_NAMESPACE

        ),
        summarizer=Summarizer.proposal_summarizer(llm=llm, verbose=False)
    )
    proposal_processor.process()


if __name__ == "__main__":
    main()
