from dash import html


class ComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_word_cloud_box(headline: str, image_id: str):
        """
        Creates a html div with headline and image
        :param headline: Headline text
        :param image_id: Image id to load
        :return: html div
        """
        return html.Div([
            html.H2(headline),
            html.Img(id=image_id)
        ])
