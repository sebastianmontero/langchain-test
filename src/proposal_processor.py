from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra, validator, root_validator, PrivateAttr
from governance_common.models import GovernanceProposal, RecordState
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging
from utils import extract_urls
from typing import List
from str_filters import BaseStrFilter, should_filter
from scrapers import BaseScarper, scrape
from typing import Optional, List
from langchain.schema import Document
from proposal_detector import BaseProposalDetector
from document_stores import BaseDocumentStore
from summarizers import BaseSummarizer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class ProposalProcessor(ABC, BaseModel):
    db_url: str
    proposal_score_threshold: float
    url_filters: List[BaseStrFilter]
    content_filters: List[BaseStrFilter]
    scrapers: List[BaseScarper]
    proposal_detector: BaseProposalDetector
    summarizer: BaseSummarizer
    document_store: BaseDocumentStore
    _session_maker: sessionmaker = PrivateAttr()


    def __init__(self, **data):
        super().__init__(**data)
        engine = create_engine(self.db_url)
        self._session_maker = sessionmaker(bind=engine)

    def process(self):
        with self._session_maker() as session:
            # proposals = GovernanceProposal.find_unprocessed(network_id="polkadot", governance_proposal_type_id="treasury_proposals", session=session)
            proposals = GovernanceProposal.get_by_id(governance_proposal_id="ad113ff6-f37a-4d4d-a811-6cb9335173a0", session=session)
        # proposals = proposals[:20]
        proposals = [proposals]
        for proposal in proposals:
            try:
              content = self._get_content(proposal)
              if content:
                  doc = Document(
                      page_content=content["text"],
                      metadata={
                          "governance_proposal_path": proposal.governance_proposal_path,
                          "governance_proposal_id": str(proposal.governance_proposal_id),
                          "source": content["source"],
                          "network": proposal.network_id,
                          "governance_proposal_type": proposal.governance_proposal_type_id,
                          "governance_proposal_title": proposal.title,
                          "chunk-context": f"{proposal.title}\nproposal path: {proposal.governance_proposal_path}",
                          "type": "proposal",
                      }
                  )
                  summary = self.summarizer.summarize(doc)
                  self.document_store.store(doc)
                  proposal.summary = summary
                  proposal.record_state_id =  RecordState.PROCESSED
            except Exception as e:
                logger.error("An error occurred: %s", str(e), exc_info=True)
                proposal.record_state_id =  RecordState.UNABLE_TO_PROCESS
            with self._session_maker() as session:
                session.merge(proposal)
                session.commit()
    
    def _get_content(self, proposal: GovernanceProposal) -> Optional[dict]:
        text = proposal.content
        title = proposal.title
        logger.debug(f"\n\nprocessing proposal: {proposal.governance_proposal_id}, path: {proposal.governance_proposal_path} title: {title}")

        if should_filter(self.content_filters, text):
            return None
        urls = self._filter_urls(extract_urls(text))
        best_proposal = None
        should_print_content = False
        for url in urls:
            logger.debug(f"\nprocessing url: {url}")
            t = scrape(self.scrapers, url)
            score = self.proposal_detector.is_proposal(title, t)
            logger.debug(f"score: {score}, url: {url}")
            should_print_content = should_print_content or score > 0.5
            if self.proposal_score_threshold <= score:
                if not best_proposal or best_proposal["score"] < score:
                    best_proposal = {
                        "score": score,
                        "source": url,
                        "text": t
                    }
        if should_print_content:
            logger.debug(f"\ncontent:\n {text}")
        if best_proposal:
            return best_proposal
        return {
            "source": proposal.governance_proposal_id,
            "text": text
        }
                    
                

    def _filter_urls(self, urls: List[str]) -> List[str]:
        filtered = []
        for url in urls:
            if not should_filter(self.url_filters, url):
                filtered.append(url)
        return filtered
