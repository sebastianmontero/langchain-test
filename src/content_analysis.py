from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from models import GovernanceProposal
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import marko

from marko import Markdown
from marko.ast_renderer import ASTRenderer

import mimetypes
import urllib.request
from urllib.parse import urlparse

import os
import re
import json


def main():
    load_dotenv()

    DB_URL = os.getenv("DB_URL")
    
    engine = create_engine(DB_URL)
    session_maker = sessionmaker(bind=engine)

    with session_maker() as session:
        gps = session.query(GovernanceProposal).filter_by(network_id="polkadot", governance_proposal_type_id="treasury_proposals")

    total_results = 0
    # text = "Reward for producing a very comprehensive article on how to participate in Acala's Early Adopter. The article was written in Portuguese and starts by explaining basic concepts about what they are: stablecoins, ausd, lcdot, Pool liquidity, Bootstrap, LP Tokens, Swap, Bridge. The article also explains what the Early Adopters Program is and what are the requirements to participate. Then the article explains, in the form of a tutorial, how to bridge dots to acala, how to mint ausd, manage a vault, provide liquidity in the pool. The article ends with a translation of the official karura FAQ. The article was highly praised in the Brazilian Polkadot and Acala community on Telegram. link to article: [https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f](https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f) ____________________________________________________ Giorge Abdala is graduated in IT from UFPR(Brazil), with a postgraduate degree in Marketing Management and an MBA in Financial Markets. He is an active member of the Brazilian DotSama community, software developer, passionate about the Polkadot ecosystem, Kusama and their parachains. He produces original content on Medium Blog PolkaMix and translates content not yet translated into Portuguese about the DotSama world in general."
    # text = "link to article: [https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f](https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f)"
    # text = "link to article: [https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f?a=hola][https://medium.com/@giorgeabdala/guia-definitivo-da-acala-saiba-tudo-o-que-precisa-para-participar-do-early-adopters-program-64882504ae6f)"
    # url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
    # url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()\[\]<>]+|\(([^\s()\[\]<>]+|(\([^\s()\[\]<>]+\)))*\))+(?:\(([^\s()\[\]<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
    # url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    # url_pattern = re.compile(r"^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")

    # print(f"{text}\n\n\n")
    # urls = re.findall(url_pattern, text)
    # urls = re.match(url_pattern, text)
    # print(urls)

    # clean_content = "This proposal is about how much XRT should be allocated from the budget to support the grants program for the development of Home Assistant use cases for the next 6 months. The proposal is part of the overall vote (the other part is presented on the Ethereum network [using](https://snapshot.org/#/developers.robonomics.eth) the [Snapshot](https://snapshot.org/#/0.robonomics.eth) [platform](https://snapshot.org/#/1.robonomics.eth)) and is intended for XRT holders on the Robonomics parachain on Kusama. Every 20 XRT which votes **aye** will add 1 XRT to the Home Assistant grant budget, every 20 XRT with **nay** will reduce the budget by 1 XRT. More information about Town Hall 2022 and voting process: [https://robonomics.network/blog/robonomics-town-hall-2022](https://robonomics.network/blog/robonomics-town-hall-2022) ### Technical details The proposal is to execute a transaction with the `system.remark` call, which contains the following string: > Proposal: Approve the allocation of XRT from the budget for Home Assistant Support Grants. The system remark was chosen because it actually does not affect any change in the state of the Robonomics chain, but allows us to find out the opinion of holders by storing the voting results in the parachain. Encoded call hash: ``` 0xd6fb1e78d642b52563c31902f5794eaad81ab77741e112cc2a65b3703f8baf0e ``` Encoded call data, which can be decoded on Robonomics parachain: ``` 0x0a01690150726f706f73616c3a20417070726f76652074686520616c6c6f636174696f6e206f66205852542066726f6d207468652062756467657420666f7220486f6d6520417373697374616e7420537570706f7274204772616e74732e ``` ### Voting process The functionality of voting is practically identical to voting on the Kusama network. First, the community chooses which proposal will be put to a referendum at the beginning of the next launch period (the duration of which can be seen on the Governance -> Democracy on the parachain portal) by endorsement of proposals with XRT. The proposal with the largest sum of endorsements wins. Then, from the next launch period, the referendum begins. The community votes either **aye** or **nay**, and with the ability to amplify their vote with more tokens. Used tokens are locked for the duration of the voting and can be returned after it ends. More about this mechanism can be found on the Polkadot wiki: [https://wiki.polkadot.network/docs/learn-governance](https://wiki.polkadot.network/docs/learn-governance)"
    # results = extract_text_around_urls(clean_content, 200, 200)
    # #    total_results += len(results)
    # # print(f"num results: {total_results}")
    # for k,v in results.items():
    #     print(f"{k}:\n{v}\n\n\n")
    #     print(f"{clean_content}\n\n\n")

    urls ={}
    for gp in gps:
       clean_content = clean_whitespace(gp.content)
       results = extract_text_around_urls(clean_content, 200, 200)
    #    total_results += len(results)
    # print(f"num results: {total_results}")
       for r in results:
        urls[get_domain(r['url'])] = True
        #    print(f"{r['url']}:\n{r['text']}\n\n\n")
        #    print(f"{clean_content}\n\n\n")
        #    print(f"{r['url']}: {get_url_mime_type(r['url'])}")
    #     # if len(v) > 530:
        #     print(f"<{k}>:\n{v}\n\n\n")
        #     print(f"{clean_content}\n\n\n")
            # print(f"{k}: {get_url_mime_type(k)}")
    
    # for gp in gps:
    #    results = get_links(gp.content)
    #    for url in results:
    #     #    print(f"{k}:\n{v}")
    #     print(f"{url}")
    print(json.dumps(urls, indent=4))
    print(len(urls))

def get_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

def clean_whitespace(text: str) -> str:
    
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()


def extract_text_around_urls(text, max_prefix, max_suffix):
    url_pattern = re.compile(r'(https?://\S+)')
    url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()\[\]<>]+|\(([^\s()\[\]<>]+|(\([^\s()\[\]<>]+\)))*\))+(?:\(([^\s()\[\]<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")


    # print(f"{text}\n\n\n")
    urls = re.findall(url_pattern, text)
    # print(f"After finding urls\n\n\n")
    url_data = []
    current_url_data = None
    search_from = 0
    for url in urls:
        url = url[0]
        # print(f"{url}\n\n\n")
        url_index = text.find(url, search_from)
        if url_index != -1:
            search_from = url_index + 1
            if current_url_data:
                current_url_data["end_limit"] = url_index
            current_url_data ={
                "url": url,
                "start_index": url_index
            }
            url_data.append(current_url_data)
    if current_url_data:
        current_url_data["end_limit"] = len(text)
    
    # print(url_data)
    extracted_text = []
    last_url_end_index = 0
    for url in url_data:
        url_index = url["start_index"]
        start = max(last_url_end_index, url_index - max_prefix)
        last_url_end_index = url_index + len(url["url"])
        end = min(url["end_limit"], last_url_end_index + max_suffix)
        extracted_text.append({"url":url["url"],"text":text[start:end]})
    return extracted_text

def get_url_mime_type(url):
    try:
        response = urllib.request.urlopen(url)
        content_type = response.headers['Content-Type']
        print(content_type)
        # mime_type, _ = mimetypes.guess_type(url)
        return content_type
    except Exception as e:
        print(f"Error retrieving MIME type: {e}")
        return None



if __name__ == "__main__":
    main()

  