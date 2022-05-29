from ast import literal_eval

import pandas as pd
import plotly.express as px
from dash import html, dcc

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
        """
        Build small div for top 10
        :return: Div with headline and graph id
        """
        return html.Div([
            html.H2('Top 10 tourist attraction combinations'),
            dcc.Graph(id='top_ten'),
        ])

    @staticmethod
    def update_top_ten(combinations, attractions, attraction_filter):
        """
        Update top 10 combinations by attractions
        :param combinations: Data of all combinations
        :param attractions: Filtered Attractions by continent/country/place filters
        :param attraction_filter: Single attraction filter
        :return: Figure showing top 10 combinations
        """
        filtered_str = "|".join(attractions)
        filtered_combinations = pd.DataFrame(columns=['itemsets', 'support'])
        for row in combinations.itertuples():
            combination = eval(row.itemsets)
            first_attraction, second_attraction = combination
            if (first_attraction in filtered_str) and (second_attraction in filtered_str):
                if attraction_filter and (
                        (first_attraction in attraction_filter) or (second_attraction in attraction_filter)):
                    filtered_combinations = filtered_combinations.append(pd.DataFrame([row], columns=row._fields),
                                                                         ignore_index=True)
                elif not attraction_filter:
                    filtered_combinations = filtered_combinations.append(pd.DataFrame([row], columns=row._fields),
                                                                         ignore_index=True)
        filtered_combinations = filtered_combinations.sort_values(by=['support'])
        if len(filtered_combinations) > 10:
            filtered_combinations = filtered_combinations[:10]
        fig = px.bar(filtered_combinations, x='itemsets', y='support', barmode='group')
        return fig
