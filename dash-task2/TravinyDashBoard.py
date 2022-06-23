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

directory = Path().resolve().parent

# Load data
main_file = pd.read_csv(str(directory) + '/crawling-and-preprocessing/content/data_prep_2805_3.csv')
combinations_file = pd.read_csv(str(directory) + '/crawling-and-preprocessing/content/combinations_2.csv')
topic_file = pd.read_csv(str(directory) + '/crawling-and-preprocessing/content/topic_model.csv')
image_filename = os.path.join(os.getcwd(), 'traviny_logo.png')

# Load stylesheet
app = Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# Create initial components
filter_builder = FilterComponentBuilder(main_file)
attraction_filter = filter_builder.build_attraction_filter()
scatter_diagram_clustering = ComponentBuilder.build_scatter()
topic_charts = ComponentBuilder.build_topic_charts(topic_file)

first_ddf_filter = filter_builder.build_dff_filter('-1')
second_ddf_filter = filter_builder.build_dff_filter('-2')
third_ddf_filter = filter_builder.build_dff_filter('-3')
fourth_ddf_filter = filter_builder.build_dff_filter('-4')

top_3_places = ComponentBuilder.build_top3_places()
top_3_countries = ComponentBuilder.build_top3_countries()
own_idea = ComponentBuilder.build_own_idea()

# Create Layout of site and refer to ids
app.layout = html.Div(children=[
    html.Img(id='traviny_logo'),
    dbc.Row([
        dbc.Col([attraction_filter]),
        dbc.Col([scatter_diagram_clustering]), dbc.Col([topic_charts])
    ]),
    dbc.Row([
        dbc.Col([first_ddf_filter, second_ddf_filter, third_ddf_filter, fourth_ddf_filter]),
        dbc.Col(dbc.Row([dbc.Col(top_3_places), dbc.Col(top_3_countries)])), dbc.Col(own_idea)
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
    dd.Output('dropdown-1', 'options'),
    dd.Output('dropdown-2', 'options'),
    dd.Output('dropdown-3', 'options'),
    dd.Output('dropdown-4', 'options'),
    dd.Input('dropdown-1', 'value'),
    dd.Input('dropdown-2', 'value'),
    dd.Input('dropdown-3', 'value'),
    dd.Input('dropdown-4', 'value'),
)
def change_filter(drop_one_filter, drop_two_filter, drop_three_filter, drop_four_filter):
    ['blub', None, None, 'blab']
    filters = [drop_one_filter, drop_two_filter, drop_three_filter, drop_four_filter]
    elements = list(filter_builder.elements)
    for filter in filters:
        if filter:
            elements = elements.remove(filter)
    options = [elements, elements, elements, elements]
    for filter, option in zip(filters, options):
        if filter:
            option.append(filter)
    return options


# Start app
if __name__ == '__main__':
    app.run_server(debug=True)
