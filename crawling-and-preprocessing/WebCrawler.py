import pandas as pd
import requests
from bs4 import BeautifulSoup


class WebCrawler:

    def __init__(self):
        pass

    def crawl_website(self):
        """
        Crawl website roughguides.com by entries in given csv file
        """
        guides = pd.read_csv('/content/DataScience2022_RoughGuides.csv');
        guides['introduction'] = None
        guides['description'] = None
        for row in guides.itertuples():
            url = row[5]
            crawled_page = requests.get(url)
            html_doc = crawled_page.content
            soup = BeautifulSoup(html_doc, 'html.parser')
            introduction = self.__get_introduction(soup)
            description = self.__get_description(soup)
            guides['introduction'].iloc[row.Index] = introduction
            guides['description'].iloc[row.Index] = description
        guides.to_csv('/content/crawled_rough_guides.csv', index=False)

    @staticmethod
    def __get_introduction(soup):
        """
        Get introduction of current soup.
        :return: introduction text of website
        """
        main = soup.find('main', {'class': ''})
        leading = main.find('p', {'class': 'leading-7'})
        return leading.parent.get_text()

    @staticmethod
    def __get_description(soup):
        """
        Get description of current soup.
        :return: description text of website
        """
        description = ''
        main = soup.find('div', {'class': 'main-container'})
        children = main.findChildren("p", recursive=True)
        for child in children:
            if ('</a>' not in str(child)) and ('class=\"mb-' not in str(child)):
                description += child.get_text()
        return description
