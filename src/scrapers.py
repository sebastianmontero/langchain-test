from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel, validator, root_validator
from urllib.parse import urlparse
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
import requests
from readers import ReaderFactory
from langchain.utils import get_from_dict_or_env
from utils import get_content_type


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class BaseScarper(ABC, BaseModel):
    
    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Wether this scraper can handle the provided url"""

    @abstractmethod
    def get_content(self, url: str) -> str:
        """Gets the text from the resource pointed by the url"""


class GoogleDocScraper(BaseScarper):
    service_account_key: Path

    @validator("service_account_key")
    def validate_service_account_key(cls, v: Any, **kwargs: Any) -> Any:
        """Validate that service_account_key exists."""
        if not v.exists():
            raise ValueError(f"service_account_key {v} does not exist")
        return v

    def can_handle(self, url: str) -> bool:
        """Wether this scraper can handle the provided url"""
        return url.find("https://docs.google.com/document/d") != -1

    def get_content(self, url: str) -> str:
        """Gets the text from the resource pointed by the url"""
        file_id = self.get_file_id(url)
        creds = service_account.Credentials.from_service_account_file(
                str(self.service_account_key), scopes=SCOPES
            )
        service = build("drive", "v3", credentials=creds)

        service.files().get(fileId=file_id, supportsAllDrives=True).execute()
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        try:
            while done is False:
                status, done = downloader.next_chunk()

        except HttpError as e:
            if e.resp.status == 404:
                print("File not found: {}".format(id))
            else:
                print("An error occurred: {}".format(e))

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
    
class FileUrlScraper(BaseScarper):

    def can_handle(self, url: str) -> bool:
        """Wether this scraper can handle the provided url"""
        content_type = get_content_type(url)
        return ReaderFactory.has_reader_for(content_type)

    def get_content(self, url: str) -> str:
        """Gets the text from the resource pointed by the url"""
        response = requests.get(url, stream=False)

        # Check if the request was successful
        if response.status_code == 200:
            reader = ReaderFactory.get_reader(response.headers['Content-Type'])
            return reader.get_content(response.content)
        else:
            response.raise_for_status()
    
    def to_export_url(self, url: str) -> str:
      url_type = self.url_type(url)
      if url_type == GDriveUrlType.VIEW_FILE:
          file_id = self.get_file_id(url)
          return f"https://drive.google.com/uc?export=download&id={file_id}"
      return url
    
class GDriveUrlType(Enum):
    VIEW_FILE = 1
    EXPORT_FILE = 2
    
class GoogleDriveFileScraper(FileUrlScraper):

    def can_handle(self, url: str) -> bool:
        """Wether this scraper can handle the provided url"""
        try:
          self.url_type(url)
          return True
        except:
            return False

    def get_content(self, url: str) -> str:
        """Gets the text from the resource pointed by the url"""
        url = self.to_export_url(url)
        return super().get_content(url)
    
    def to_export_url(self, url: str) -> str:
      url_type = self.url_type(url)
      if url_type == GDriveUrlType.VIEW_FILE:
          file_id = self.get_file_id(url)
          return f"https://drive.google.com/uc?export=download&id={file_id}"
      return url


    def get_file_id(self, url: str) -> str:
      """Gets the file ID from a Google Doc URL.

      Args:
        url: The Google Doc URL.

      Returns:
        The file ID.
      """

      parts = urlparse(url)
      return parts.path.split('/')[3]
    
    def url_type(self, url: str) -> GDriveUrlType:
      if url.find('drive.google.com/file') != -1:
          return GDriveUrlType.VIEW_FILE
      elif url.find('drive.google.com/uc?export') != -1:
          return GDriveUrlType.EXPORT_FILE
      raise ValueError(f"Unknown google drive url type: {url}")
    
class WebPageScraper(BaseScarper):
    
    """Wrapper around Apify.

    To use, you should have the ``apify-client`` python package installed,
    and the environment variable ``APIFY_API_TOKEN`` set with your API key, or pass
    `apify_api_token` as a named parameter to the constructor.
    """

    apify_client: Any
    apify_client_async: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate environment.

        Validate that an Apify API token is set and the apify-client
        Python package exists in the current environment.
        """
        apify_api_token = get_from_dict_or_env(
            values, "apify_api_token", "APIFY_API_TOKEN"
        )

        try:
            from apify_client import ApifyClient, ApifyClientAsync

            values["apify_client"] = ApifyClient(apify_api_token)
            values["apify_client_async"] = ApifyClientAsync(apify_api_token)
        except ImportError:
            raise ValueError(
                "Could not import apify-client Python package. "
                "Please install it with `pip install apify-client`."
            )

        return values

    def can_handle(self, url: str) -> bool:
        content_type = get_content_type(url)
        return content_type.find("text/html") != -1        

    def get_content(self, url: str) -> str:
        """Gets the text from the resource pointed by the url"""
        actor_call = self.apify_client.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": url}],
                "maxCrawlPages": 1,
                },
            # build=build,
            # memory_mbytes=memory_mbytes,
            # timeout_secs=timeout_secs,
        )

        items = self.apify_client.dataset(actor_call["defaultDatasetId"]).list_items().items
        text = ""
        for item in items:
            text += item["text"] + "\n\n"
        return text

