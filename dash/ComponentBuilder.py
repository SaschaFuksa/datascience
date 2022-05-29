from ast import literal_eval

import pandas as pd
import plotly.express as px
from dash import html, dcc


class ComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_word_cloud_box(headline: str, image_id: str):
        '''
        Creates a html div with headline and image
        :param headline: Headline text
        :param image_id: Image id to load
        :return: html div
        '''
        return html.Div([
            html.H2(headline),
            html.Img(id=image_id)
        ])

    @staticmethod
    def build_top_ten():
        '''
        Build small div for top 10
        :return: Div with headline and graph id
        '''
        return html.Div([
            html.H2('Top 10 tourist attraction combinations'),
            dcc.Graph(id='top_ten'),
        ])

    @staticmethod
    def build_world_map(main_file):
        """
        Create static world map and show sum of country ne
        :param main_file: File to extract data
        :return: Div with headline and world map graph
        """
        world_map_df = main_file[['country', 'freq_NE_int']]
        world_map_df['count_ne'] = ' '
        for freq, row in zip(world_map_df['freq_NE_int'].apply(literal_eval), world_map_df.itertuples()):
            len_freq = len(freq)
            world_map_df['count_ne'].iloc[row.Index] = len_freq
        world_map_df['country'] = world_map_df['country'].replace(
            ['Canada', 'Brazil', 'Mexico', 'Peru', 'Deutschland', 'Frankreich', 'Spanien', 'Schweden', 'Italien'],
            ['CAN', 'BRA', 'MEX', 'PER', 'DEU', 'FRA', 'ESP', 'SWE', 'ITA'])
        world_map_df = world_map_df.groupby(['country']).sum()
        world_map_df = world_map_df.reset_index(level=0)
        fig = px.choropleth(world_map_df, locations='country',
                            color='count_ne', color_continuous_scale=px.colors.sequential.PuRd)
        return html.Div([
            html.H2('World Map - sum of named entities'),
            dcc.Graph(id='world_map', figure=fig),
        ])

    @staticmethod
    def update_top_ten(combinations, places_list, attraction_filter):
        '''
        Update top 10 combinations by attractions
        :param combinations: Data of all combinations
        :param places_list: Filtered Places
        :param attraction_filter: Single attraction filter
        :return: Figure showing top 10 combinations
        '''
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
