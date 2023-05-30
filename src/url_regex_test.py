import re

text = "https://i.ibb.co/tm5CyPZ/pitchdeck.png)](https://ibb.co/r7Jynfb)"

url_pattern = re.compile(r"(https?://\S+)")
urls = re.findall(url_pattern, text)

print(urls)