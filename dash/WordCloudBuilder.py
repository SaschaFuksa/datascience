import base64
from io import BytesIO

from wordcloud import WordCloud


class WordCloudBuilder:

    def __init__(self):
        pass

    @staticmethod
    def create_word_cloud(data: str):
        """
        Create a word cloud by creating a image and decode it for dash
        :param data: Data/text to extract words
        :return: Formated base64 decoded image for word clouds
        """
        image = BytesIO()
        WordCloudBuilder.generate_word_cloud_image(data).save(image, format='PNG')
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
