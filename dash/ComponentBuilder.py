import pandas as pd
import plotly.express as px
from dash import html, dcc

''


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

    @staticmethod
    def build_top_ten():
        return html.Div([
            html.H2('Top 10 tourist attraction combinations'),
            dcc.Graph(id='top_ten'),
        ])

    @staticmethod
    def update_top_ten(combinations, attractions, attraction_filter):
        filtered_str = "|".join(attractions)
        pre_filtered_combinations = combinations[combinations['combination'].str.contains(filtered_str)]
        filtered_combinations = pd.DataFrame(columns=['combination', 'freq'])
        for row in pre_filtered_combinations.itertuples():
            combination = row.combination
            first_attraction = combination.split()[0]
            second_attraction = combination.split()[1]
            if first_attraction in filtered_str and second_attraction in filtered_str:
                filtered_combinations = filtered_combinations.append(pd.DataFrame([row], columns=row._fields),
                                                                     ignore_index=True)
        if attraction_filter:
            # todo: Filter wenn einzelne ausgewählt sind
            pass
        filtered_combinations = filtered_combinations.sort_values(by=['freq'])
        if len(filtered_combinations) > 10:
            filtered_combinations = filtered_combinations[:10]
        fig = px.bar(filtered_combinations, x='combination', y='freq', barmode='group')
        return fig
