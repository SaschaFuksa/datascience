import dash_bootstrap_components as dbc
from dash import html


class FilterComponentBuilder:

    def __init__(self, main_file):
        self.main_file = main_file

    def build_country_filter(self):
        countries = set()
        for country in self.main_file['country']:
            if country not in countries:
                countries.add(country)
        radio = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(countries)],
            id='country_selection',
            switch=True,
            style={
                'overflowY': 'scroll',
                'max-height': '120px'
            }
        )
        return html.Div([
            html.H3('Country filter'),
            radio,
        ],
            id='country_filter')

    def build_place_filter(self):
        places_to_show = self.main_file['place'].tolist()
        check_list = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(places_to_show)],
            id='place_selection',
            switch=True,
            style={
                'overflowY': 'scroll',
                'max-height': '120px'
            }
        )
        return html.Div([
            html.H3('Location filter'),
            check_list
        ],
            id='place_filter')

    def update_place_options(self, values):
        if values:
            places_to_show = self.main_file.loc[self.main_file['country'].isin(values), 'place'].tolist()
        else:
            places_to_show = self.main_file['place'].tolist()
        return [{'label': i, 'value': i} for i in sorted(places_to_show)]
