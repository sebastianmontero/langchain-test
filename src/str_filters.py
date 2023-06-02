import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel, validator
from typing import List


logger = logging.getLogger(__name__)

class BaseStrFilter(ABC, BaseModel):
    
    @abstractmethod
    def filter(self, s: str) -> bool:
        """Wether to filter the provided string"""

    
class StrContainsFilter(BaseStrFilter):
    contents_to_filter: List

    @validator('contents_to_filter')
    def validate_domains_to_filter(cls, contents_to_filter):
        if len(contents_to_filter) <= 0:
            raise ValueError("At least one content to filter has to be provided")
        return contents_to_filter

    def filter(self, s: str) -> bool:
        s = s.lower()
        for content_to_filter in self.contents_to_filter:
            if s.lower().find(content_to_filter) != -1:
                logger.debug(f"string: \n {s}\n contains:{content_to_filter} so it was filtered")
                return True
        return False
    

def should_filter(filters: List[BaseStrFilter], s: str) -> bool:
    for filter in filters:
          if filter.filter(s):
              return True
    return False