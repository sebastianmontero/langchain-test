from abc import ABC, abstractmethod
from pydantic import BaseModel, validator,  PrivateAttr
from pathlib import Path
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from typing import Any
from urllib.parse import urlparse

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class GoogleDocClient(BaseModel, ABC):
    service_account_key: Path
    _service: Any = PrivateAttr()

    @validator("service_account_key")
    def validate_service_account_key(cls, v: Any, **kwargs: Any) -> Any:
        """Validate that service_account_key exists."""
        if not v.exists():
            raise ValueError(f"service_account_key {v} does not exist")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        creds = service_account.Credentials.from_service_account_file(
                str(self.service_account_key), scopes=SCOPES
            )
        self._service = build("drive", "v3", credentials=creds)
        
    def get_content_type(self, url: str) -> str:
        file_id = self.get_file_id(url)
        return self.get_content_type_by_file_id(file_id)
    
    def get_content_type_by_file_id(self, file_id: str) -> str:
        file = self._service.files().get(fileId=file_id).execute()
        print(file)
        return file['mimeType']
    
    def download_file(self, url: str) -> BytesIO:
        file_id = self.get_file_id(url)
        request = self._service.files().get_media(fileId=file_id)
        file = BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f'Download {int(status.progress() * 100)}.')
        return file
    
    def get_content(self, url: str) -> str:
        file_id = self.get_file_id(url)

        self._service.files().get(fileId=file_id, supportsAllDrives=True).execute()
        request = self._service.files().export_media(fileId=file_id, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        text = fh.getvalue().decode("utf-8")
        return text
    
    def get_file_id(self, url: str) -> str:
      """Gets the file ID from a Google Doc URL.

      Args:
        url: The Google Doc URL.

      Returns:
        The file ID.
      """

      parts = urlparse(url)
      return parts.path.split('/')[3]
    