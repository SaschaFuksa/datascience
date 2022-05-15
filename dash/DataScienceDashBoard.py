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

country_filter = FilterComponentBuilder.build_country_filter(main_file['country'])
place_filter = FilterComponentBuilder.build_place_filter(main_file['place'])

app.layout = html.Div(children=[
    html.H1(children='Data Science Travel Dashboard'),
    dbc.Row([
        dbc.Col([named_entity_word_cloud]), dbc.Col([activity_word_cloud])
    ]),
    dbc.Row([
        dbc.Col([country_filter]), dbc.Col([place_filter])
    ]),
])


@app.callback(dd.Output('image_named_entity_word_cloud', 'src'), [dd.Input('image_named_entity_word_cloud', 'id')])
def make_image(b):
    return WordCloudBuilder().create_word_cloud(example_fruits)


@app.callback(dd.Output('image_activity_word_cloud', 'src'), [dd.Input('image_activity_word_cloud', 'id')])
def make_image(b):
    return WordCloudBuilder().create_word_cloud(example_vegetables)


if __name__ == '__main__':
    app.run_server(debug=True)
