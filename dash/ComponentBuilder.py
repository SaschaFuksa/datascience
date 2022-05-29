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
    def update_top_ten(combinations, places_list, attraction_filter):
        """
        Update top 10 combinations by attractions
        :param combinations: Data of all combinations
        :param places_list: Filtered Places
        :param attraction_filter: Single attraction filter
        :return: Figure showing top 10 combinations
        """
        valid_places_combination_df = pd.DataFrame(columns=['support', 'itemsets', 'place'])
        for place in places_list:
            valid_places_combination_df = valid_places_combination_df.append(
                combinations.loc[combinations['place'] == place])
        if attraction_filter:
            places_combination_df = pd.DataFrame(columns=['support', 'itemsets', 'place'])
            for row in valid_places_combination_df.itertuples():
                if any(word_item in row.itemsets.split(' ') for word_item in attraction_filter):
                    places_combination_df = places_combination_df.append(
                        {'support': row.support, 'itemsets': row.itemsets, 'place': row.place},
                        ignore_index=True)
        else:
            places_combination_df = valid_places_combination_df
        filtered_combinations = places_combination_df.sort_values(by=['support'], ascending=False)
        if len(filtered_combinations) > 10:
            filtered_combinations = filtered_combinations[:10]
        fig = px.bar(filtered_combinations, x='itemsets', y='support', barmode='group')
        return fig
