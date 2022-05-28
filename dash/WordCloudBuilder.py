import base64
from io import BytesIO

import pandas as pd
from wordcloud import WordCloud


class WordCloudBuilder:

    def __init__(self):
        pass

    @staticmethod
    def create_word_cloud(places_list, data):
        """
        Create a word cloud by creating a image and decode it for dash
        :param places_list: List of relevant places
        :param data: Data/text to extract words
        :return: Formated base64 decoded image for word clouds
        """
        nes_frame = data.loc[data['place'].isin(places_list), ['word', 'freq']]
        nes_list = []
        freq_list = []
        for nes in nes_frame['word'].tolist():
            nes_list.extend(nes)
        for freq in nes_frame['freq'].tolist():
            freq_list.extend(freq)
        nes_wc_df = pd.DataFrame({'word': nes_list, 'freq': freq_list})
        nes_wc_df = nes_wc_df.sort_values(by=['freq'])
        if len(nes_wc_df) > 100:
            nes_wc_df = nes_wc_df[:100]
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
        word_cloud = WordCloud(background_color='white', width=480, height=360)
        word_cloud.fit_words(words)
        return word_cloud.to_image()
