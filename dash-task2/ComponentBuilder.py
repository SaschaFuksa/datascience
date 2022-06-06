from dash import html

class ComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_scatter():
        return html.Div([
            html.H2('Attraction clusters'),
        ])

    @staticmethod
    def build_word_cloud():
        return html.Div([
            html.H2('Topic word cloud'),
        ])

    @staticmethod
    def build_top3_places():
        return html.Div([
            html.H2('Top 3 places'),
        ])

    @staticmethod
    def build_top3_countries():
        return html.Div([
            html.H2('Top 3 countries'),
        ])

    @staticmethod
    def build_own_idea():
        return html.Div([
            html.H2('Own idea'),
        ])
