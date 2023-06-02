from abc import ABC, abstractmethod
from pydantic import BaseModel, validator
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Union
import PyPDF2
import docx



class ReaderFactory():

    @classmethod
    def get_reader(cls, content_type: str) -> "BaseReader":
        if cls.has_reader_for(content_type):
            return READERS[content_type]()
        else:
            raise ValueError(f"There is no reader for mime type: {content_type}")
        
    @classmethod
    def has_reader_for(cls, content_type: str) -> bool:
        return content_type in READERS



class BaseReader(ABC, BaseModel):

    @abstractmethod
    def get_content(self, data: BytesIO) -> str:
        """Gets the text from the resource"""


class PdfReader(BaseReader):

    def get_content(self, data: BytesIO) -> str:
        """Gets the text from the resource"""
        pdf_reader = PyPDF2.PdfReader(data)        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text
    
class WordDocReader(BaseReader):

    def get_content(self, data: BytesIO) -> str:
        """Gets the text from the resource"""
        doc = docx.Document(data)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    
READERS = {
    "application/pdf": PdfReader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": WordDocReader
}
