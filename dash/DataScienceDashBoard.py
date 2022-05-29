import base64
import os
from pathlib import Path

import dash
import dash.dependencies as dd
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html

from ComponentBuilder import ComponentBuilder
from FilterComponentBuilder import FilterComponentBuilder
from WordCloudBuilder import WordCloudBuilder

directory = Path().resolve().parent

# Load data
example_combinations = pd.DataFrame({'combination': ['bars club', 'club restaurant', 'beach club', 'beach restaurant'],
                                     'freq': [4, 5, 1, 2]})
main_file = pd.read_csv(str(directory) + '/crawling-and-preprocessing/content/data_prep_2805_2.csv')
image_filename = os.path.join(os.getcwd(), 'traviny_logo.png')

# Load stylesheet
app = Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# Create initial components
named_entity_word_cloud = ComponentBuilder.build_word_cloud_box('Named Entity Word Cloud',
                                                                'image_named_entity_word_cloud')
activity_word_cloud = ComponentBuilder.build_word_cloud_box('Activity Word Cloud',
                                                            'image_activity_word_cloud')
top_10_attractions = ComponentBuilder.build_top_ten()
continent_filter = FilterComponentBuilder.build_filter(main_file, 'continent', 'Continent filter')
country_filter = FilterComponentBuilder.build_filter(main_file, 'country', 'Country filter')
place_filter = FilterComponentBuilder.build_filter(main_file, 'place', 'Place filter')
attraction_filter = FilterComponentBuilder.build_filter(main_file, 'non_NE_nouns', 'Attraction filter')

# Create Layout of site and refer to ids
app.layout = html.Div(children=[
    html.Img(id='traviny_logo'),
    dbc.Row([
        dbc.Col([continent_filter, country_filter, place_filter]),
        dbc.Col([named_entity_word_cloud]), dbc.Col([activity_word_cloud])
    ]),
    dbc.Row([
        dbc.Col([attraction_filter]),
        dbc.Col(top_10_attractions), dbc.Col(html.H2('Own idea'))
    ]),
])


@app.callback(dd.Output('traviny_logo', 'src'), [dd.Input('traviny_logo', 'id')])
def make_logo_image(id):
    """
    Insert Logo on page
    :param id: To recognize img area
    :return: Decoded image
    """
    encoded = base64.b64encode(open('traviny_logo.png', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


@app.callback(
    dd.Output('country_selection', 'options'),
    dd.Output('place_selection', 'options'),
    dd.Output('non_NE_nouns_selection', 'options'),
    dd.Output('image_named_entity_word_cloud', 'src'),
    dd.Output('image_activity_word_cloud', 'src'),
    dd.Output('top_ten', 'figure'),
    dd.Input('continent_selection', 'value'),
    dd.Input('country_selection', 'value'),
    dd.Input('image_named_entity_word_cloud', 'id'),
    dd.Input('image_activity_word_cloud', 'id'),
    dd.Input('place_selection', 'value'),
    dd.Input('non_NE_nouns_selection', 'value')
)
def change_filter(continent_filter, country_filter, ne_id, att_id, place_filter, attraction_filter):
    """
    Handle callback for all filter actions
    :param continent_filter: List of filtered continents, empty if none selected
    :param country_filter: List of filtered counties, empty if none selected
    :param ne_id: id of ne wc
    :param att_id: id of attraction wc
    :param place_filter: List of filtered places, empty if none selected
    :param attraction_filter: List of filtered attractions, empty if none selected
    :return: All needed updated options and contents (wordcloud images and figures)
    """
    countries, places, places_list, attractions, attractions_list = FilterComponentBuilder.update_options(main_file,
                                                                                                          continent_filter,
                                                                                                          country_filter,
                                                                                                          place_filter)
    if place_filter:
        places_list = places_list.intersection(place_filter)
    wc_ne = WordCloudBuilder().create_word_cloud(places_list, main_file, 'NE_no_tag', 'freq_NE_int')
    wc_att = WordCloudBuilder().create_word_cloud(places_list, main_file, 'non_NE_nouns', 'freq_noun_int')
    fig = ComponentBuilder.update_top_ten(example_combinations, attractions_list, attraction_filter)
    return countries, places, attractions, wc_ne, wc_att, fig


# Start app
if __name__ == '__main__':
    app.run_server(debug=True)
