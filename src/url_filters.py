from utils import get_content_type, get_domain, to_dict
from abc import ABC, abstractmethod, abstractclassmethod
from pydantic import BaseModel, validator, root_validator
from typing import List


class BaseUrlFilter(ABC, BaseModel):
    
    @abstractmethod
    def filter(self, url: str) -> bool:
        """Wether to filter the provided url"""

class DomainUrlFilter(ABC, BaseModel):
    domains_to_filter: List

    @validator('domains_to_filter')
    def validate_domains_to_filter(cls, domains_to_filter):
        if len(domains_to_filter) <= 0:
            raise ValueError("At least one domain to filter has to be provided")
        return domains_to_filter
    
    def __init__(self, **data):
        super().__init__(**data)
        self.domains = to_dict(self.domains_to_filter)


    def filter(self, url: str) -> bool:
        return get_domain(url) in self.domains
    
class ContentTypeFilter(ABC, BaseModel):
    content_types_to_filter: List

    @validator('content_types_to_filter')
    def validate_content_types_to_filter(cls, content_types_to_filter):
        if len(content_types_to_filter) <= 0:
            raise ValueError("At least one content type to filter has to be provided")
        return content_types_to_filter
    
    def __init__(self, **data):
        super().__init__(**data)
        self.content_types = to_dict(self.content_types_to_filter)


    def filter(self, url: str) -> bool:
        content_type = get_content_type(url)
        if content_type in self.content_types:
            return True
        return content_type.split("/")[0] in self.content_types