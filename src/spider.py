import scrapy

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://centrifuge.hackmd.io/Q4AZOW2WRPq7Q0Ti3ee8Og']

    def parse(self, response):
        div_content = response.css('div#doc::text').get()
        yield {'div_content': div_content}