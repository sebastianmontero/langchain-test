
import requests
from urllib.parse import urlparse
import re
from typing import Set

def get_content_type(url: str) -> str:
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        return response.headers['Content-Type']
    else:
        response.raise_for_status()

def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_plural(token: str) -> str:
      if token.endswith('s'):
          return token[:-1]
      return token

def to_dict(l: list, d: dict = None) -> dict:
    if not d:
        d = {}
    for v in l:
        d[v] = True
    return d

def extract_urls(text: str) -> Set[str]:
    url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()\[\]<>]+|\(([^\s()\[\]<>]+|(\([^\s()\[\]<>]+\)))*\))+(?:\(([^\s()\[\]<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
    return set(map(lambda url: url[0], re.findall(url_pattern, text)))