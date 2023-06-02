from utils import get_content_type, get_domain, to_dict
from pydantic import validator, PrivateAttr
from typing import List, Dict
from str_filters import BaseStrFilter, StrContainsFilter
from requests import HTTPError
from requests.exceptions import SSLError
import logging


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class DomainUrlFilter(BaseStrFilter):
    domains_to_filter: List[str]
    _domains: Dict[str, bool] = PrivateAttr()
    @validator('domains_to_filter')
    def validate_domains_to_filter(cls, domains_to_filter):
        if len(domains_to_filter) <= 0:
            raise ValueError("At least one domain to filter has to be provided")
        return domains_to_filter
    
    def __init__(self, **data):
        super().__init__(**data)
        self._domains = to_dict(self.domains_to_filter)


    def filter(self, url: str) -> bool:
        return get_domain(url) in self._domains
    
error_filter = StrContainsFilter(contents_to_filter=[
                "410 client error: gone for url",
                "404 client error: not found for url"
            ])
class ContentTypeFilter(BaseStrFilter):
    content_types_to_filter: List[str]
    _content_types: Dict[str, bool] = PrivateAttr()
    @validator('content_types_to_filter')
    def validate_content_types_to_filter(cls, content_types_to_filter):
        if len(content_types_to_filter) <= 0:
            raise ValueError("At least one content type to filter has to be provided")
        return content_types_to_filter
    
    def __init__(self, **data):
        super().__init__(**data)
        self._content_types = to_dict(self.content_types_to_filter)


    def filter(self, url: str) -> bool:
        try:
            logger.debug(f"Checking content type of: {url}")
            content_type = get_content_type(url)
            logger.debug(f"Content type of: {url}, {content_type}")
            if content_type in self._content_types:
                return True
            return content_type.split("/")[0] in self._content_types
        except HTTPError as e:
            
            if error_filter.filter(e.__str__()):
                logger.debug(f"filtering error: {e}")
                return True
            raise e
        except SSLError as e:
            return True