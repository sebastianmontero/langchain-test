from abc import ABC, abstractmethod
from collections import Counter
from pydantic import BaseModel, validator, root_validator
from typing import Any, Dict, List, Optional, Sequence, Union
from utils import clean_text, remove_plural, to_dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk

nltk.download('punkt')


class BaseProposalDetector(ABC, BaseModel):
    
    @abstractmethod
    def is_proposal(self, title: str, text: str) -> bool:
        """Wether the document is a proposal"""


class KeywordProposalDetector(ABC, BaseModel):
    keywords: set = {'proposal', 'scope', 'objective', 'budget', 'deliverable', 'milestone'}
    keywords_threshold: float = 0.7
    title_words_threshold: float = 0.7
    
    def is_proposal(self, title: str, document: str) -> bool:
        """Wether this scraper can handle the provided url"""
        title_tokens = set(self.tokenize(title))
        doc_tokens = self.tokenize(document)
        kwords = to_dict(title_tokens)
        kwords = to_dict(self.keywords, kwords)
        matches = Counter(token for token in doc_tokens if token in kwords)  
        keywords_score = self._score(matches, self.keywords)
        if keywords_score < self.keywords_threshold:
           return False
        title_words_score = self._score(matches, self.title_tokens)
        return title_words_score >= self.title_words_threshold
        
    
    def _score(self, matches: Counter, kwords: list) -> float:
      count = 0
      for kw in kwords:
        if kw in matches:
           count += 1    
      
      return count / len(kwords)

    def _tokenize(self, text: str) -> List:
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = clean_text(text)
        tokens = word_tokenize(text)

        return [remove_plural(token) for token in tokens if token not in stop_words]