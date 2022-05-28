import base64
import operator
from ast import literal_eval
from collections import Counter
from io import BytesIO
from itertools import islice

import pandas as pd
from numpy import take
from wordcloud import WordCloud
import re

class WordCloudBuilder:

    def __init__(self):
        pass

    @staticmethod
    def create_word_cloud(places_list, data, word_col:str, freq_col: str):
        """
        Create a word cloud by creating a image and decode it for dash
        :param places_list: List of relevant places
        :param data: Data/text to extract words
        :return: Formated base64 decoded image for word clouds
        """
        nes_frame = data.loc[data['place'].isin(places_list), [word_col, freq_col]]
        nes_list = []
        freq_list = []
        for nes in nes_frame[word_col].apply(literal_eval):
            nes_list.extend(nes)
        for freq in nes_frame[freq_col].apply(literal_eval):
            freq_list.extend(map(int, freq))
        nes_wc_df = pd.DataFrame({word_col: nes_list, freq_col: freq_list})
        nes_wc_df = nes_wc_df.sort_values(by=[freq_col], ascending=False)
        image = BytesIO()
        WordCloudBuilder.generate_word_cloud_image(nes_wc_df).save(image, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(image.getvalue()).decode())

    @staticmethod
    def generate_word_cloud_image(data: str):
        """
        Generates a word cloud image of given input text
        :param data: Data to extract words
        :return: Generated image of word cloud
        """
        words = {a: x for a, x in data.values}
        if len(words) > 100:
            words = dict(sorted(words.items(), key = operator.itemgetter(1), reverse = True)[:100])
        word_cloud = WordCloud(background_color='white', width=480, height=360)
        word_cloud.fit_words(words)
        return word_cloud.to_image()
