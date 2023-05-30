from marko.block import Heading
from marko.inline import RawText
import marko
import json

# The markdown string.
markdown_text = """
[https://robonomics.network/blog/robonomics-town-hall-2022](https://robonomics.network/blog/robonomics-town-hall-2022)
https://google.com

This is some markdown text.

It has some headings:

## Heading 1

### Heading 2

And some paragraphs:

This is a paragraph.

This is another paragraph.
"""

# Parse the markdown text.
doc = marko.parse(markdown_text)

# Print the HTML text.
print(json.dumps(doc, indent=4, default=lambda o: o.__dict__))
print(doc.children[1])
print(doc.children[1].children[0].children)