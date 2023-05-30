from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
import io
import requests
import mimetypes
import urllib.request
from urllib.parse import urlparse
from enum import Enum

from PyPDF2 import PdfReader
import os
import re
import json
from langchain.document_loaders import GoogleDriveLoader
from scrapers import GoogleDocScraper, GoogleDriveFileScraper, WebPageScraper


class GDriveUrlType(Enum):
    VIEW_FILE = 1
    EXPORT_FILE = 2


def main():
    load_dotenv()
    # print(f"google doc: {get_url_mime_type('https://docs.google.com/document/d/1iYOIe_pyOdnV27hUA6FToWVSs7uUNHxIm5bA3KrMAE8/edit')}")
    # print(f"pdf: {get_url_mime_type('https://drive.google.com/file/d/1nGdzd8Pyc-er0qIiON_K3EAiQzvQjKCz/view')}")
    # result = get_file_from_drive(f"https://drive.google.com/file/d/1nGdzd8Pyc-er0qIiON_K3EAiQzvQjKCz/view")
    # text = extract_text_from_pdf(result["contents"])
    # extract_text_from_google_doc("https://docs.google.com/document/d/1iYOIe_pyOdnV27hUA6FToWVSs7uUNHxIm5bA3KrMAE8/edit")
    extract_text_from_google_drive_file("https://drive.google.com/file/d/1nGdzd8Pyc-er0qIiON_K3EAiQzvQjKCz/view")
    # extract_text_from_web_page("https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/google_drive.html")
    # print(text)

def extract_text_from_web_page(url):
    scraper = WebPageScraper()
    if not scraper.can_handle(url):
        raise ValueError(f"Web Scraper can not handle url: {url}")
    print(scraper.get_content(url))

def extract_text_from_google_doc(url):
    scraper = GoogleDocScraper(
        service_account_key="/home/sebastian/Documents/hashed/gcloud/keys/iasc-annular-form-334721-c28badd5f7aa.json",
    )
    if not scraper.can_handle(url):
        raise ValueError(f"Google Doc Scraper can not handle url: {url}")
    
    print(scraper.get_content(url))

def extract_text_from_google_drive_file(url):
    scraper = GoogleDriveFileScraper()
    if not scraper.can_handle(url):
        raise ValueError(f"Google Drive Scraper can not handle url: {url}")
    
    print(scraper.get_content(url))

# def extract_text_from_google_doc(url):
#     loader = GoogleDriveLoader(
#         # service_account_key="/home/sebastian/Documents/hashed/gcloud/langchain_oauth_credentials.json",
#         service_account_key="/home/sebastian/Documents/hashed/gcloud/keys/iasc-annular-form-334721-c28badd5f7aa.json",
#         # credentials_path="/home/sebastian/Documents/hashed/gcloud/governance-ai-gdrive-credentials.json",
#         document_ids=["1iYOIe_pyOdnV27hUA6FToWVSs7uUNHxIm5bA3KrMAE8"]
#         # file_ids=["1nGdzd8Pyc-er0qIiON_K3EAiQzvQjKCz"]
#     )
#     docs = loader.load()
#     print(docs)
    


def get_file_from_drive(url):
    
    url = to_gdrive_export_url(url)
    response = requests.get(url, stream=False)

    # Check if the request was successful
    if response.status_code == 200:
        # print(response.headers['Content-Type'])
        # print(response.content)
        return {
            "type": response.headers['Content-Type'],
            "contents": response.content,
        }
    else:
        response.raise_for_status()

def to_gdrive_export_url(url: str) -> str:
    url_type = gdrive_url_type(url)
    print(url_type)
    if url_type == GDriveUrlType.VIEW_FILE:
        file_id = get_gdrive_file_id(url)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def get_gdrive_file_id(url):
  """Gets the file ID from a Google Drive URL.

  Args:
    url: The Google Drive URL.

  Returns:
    The file ID.
  """

  # Split the URL into its components.
  parts = urlparse(url)
#   print(url)
#   print(parts)
  # Get the file ID from the query string.
  file_id = parts.path.split('/')[3]

  # Return the file ID.
  return file_id

def gdrive_url_type(url: str) -> GDriveUrlType:
    if url.find('drive.google.com/file') != -1:
        return GDriveUrlType.VIEW_FILE
    elif url.find('drive.google.com/uc?export') != -1:
        return GDriveUrlType.EXPORT_FILE
    raise ValueError(f"Unknown google drive url type: {url}")

def is_gdrive_view_file_url(url: str) -> bool:
    url.find('drive.google.com/file') != -1

def extract_text_from_pdf(contents) -> str:
    print(type(contents))
    pdf_reader = PdfReader(io.BytesIO(contents))
    # Get the number of pages in the PDF file
    
    text = ""
    # Loop through the pages in the PDF file
    for page in pdf_reader.pages:

        # Get the text on the current page
        text += page.extract_text()

        # Do something with the text on the current page
    return text


def get_url_mime_type(url):
    try:
        response = urllib.request.urlopen(url)
        content_type = response.headers['Content-Type']
        # print(content_type)
        # mime_type, _ = mimetypes.guess_type(url)
        return content_type
    except Exception as e:
        print(f"Error retrieving MIME type: {e}")
        return None



if __name__ == "__main__":
    main()

  