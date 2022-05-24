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
main_file = pd.read_csv(str(directory) + '/crawling-and-preprocessing/content/crawled_rough_guides.csv')

app = Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

example_fruits = pd.DataFrame({'word': ['apple', 'pear', 'orange'], 'freq': [1, 3, 9]})
example_vegetables = pd.DataFrame({'word': ['broccoli', 'onion', 'garlic'], 'freq': [4, 8, 20]})

named_entity_word_cloud = ComponentBuilder.build_word_cloud_box('Named Entity Word Cloud',
                                                                'image_named_entity_word_cloud')
activity_word_cloud = ComponentBuilder.build_word_cloud_box('Activity Word Cloud',
                                                            'image_activity_word_cloud')

filter_component_builder = FilterComponentBuilder(main_file)
continent_filter = filter_component_builder.build_continent_filter()
country_filter = filter_component_builder.build_country_filter()
place_filter = filter_component_builder.build_place_filter()

image_filename = os.path.join(os.getcwd(), 'traviny_logo.png')

app.layout = html.Div(children=[
    html.Img(id='traviny_logo'),
    dbc.Row([
        dbc.Col([continent_filter, country_filter, place_filter]),
        dbc.Col([named_entity_word_cloud]), dbc.Col([activity_word_cloud])
    ]),
    dbc.Row([
        dbc.Col([]),
        dbc.Col(html.H2('Top 10 tourist attraction combinations')), dbc.Col(html.H2('Own idea'))
    ]),
])


@app.callback(dd.Output('image_named_entity_word_cloud', 'src'), [dd.Input('image_named_entity_word_cloud', 'id')])
def make_image(b):
    return WordCloudBuilder().create_word_cloud(example_fruits)


@app.callback(dd.Output('image_activity_word_cloud', 'src'), [dd.Input('image_activity_word_cloud', 'id')])
def make_image(b):
    return WordCloudBuilder().create_word_cloud(example_vegetables)


@app.callback(dd.Output('traviny_logo', 'src'), [dd.Input('traviny_logo', 'id')])
def make_image(b):
    encoded = base64.b64encode(open('traviny_logo.png', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


@app.callback(
    dd.Output('country_selection', 'options'),
    dd.Output('place_selection', 'options'),
    dd.Input('continent_selection', 'value'),
)
def change_filter(value):
    return filter_component_builder.update_country_options(value), filter_component_builder.update_place_options_by_continent(value)


@app.callback(
    dd.Output('place_selection', 'options'),
    dd.Input('country_selection', 'value'),
)
def change_filter(value):
    return filter_component_builder.update_place_options(value)


if __name__ == '__main__':
    app.run_server(debug=True)
