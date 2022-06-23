from ast import literal_eval

import dash_bootstrap_components as dbc
from dash import html, dcc


class FilterComponentBuilder:

    def __init__(self, main_file):
        elements = []
        attrs_lists = main_file['singular_cleaned_nouns'].apply(literal_eval)
        for attrs_list in attrs_lists:
            for attraction in attrs_list:
                elements.append(attraction)
        self.elements = set(elements)

    def build_attraction_filter(self):
        """
        Build a new filter composite
        :param main_file: File to use for initial data
        :param column_name: name of column -> will also be prefix of ids
        :param headline: Headline of this filter component
        :return: A div with headline and options
        """
        radio_buttons = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(self.elements)],
            id='attractions_selection',
            switch=True,
            style={
                'overflowY': 'scroll',
                'max-height': '120px'
            }
        )
        html_filter = html.Div([html.H3('Attraction filter'), radio_buttons], id='attractions_filter')
        return html_filter

    def build_dff_filter(self, number: str):
        return html.Div([
            html.H2('DDF ' + number + ' filter'),
            dcc.Dropdown(options=list(self.elements), id='dropdown' + number),
        ])
