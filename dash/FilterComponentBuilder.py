import dash_bootstrap_components as dbc
from dash import html


class FilterComponentBuilder:

    def __init__(self, main_file):
        self.main_file = main_file

    def build_continent_filter(self):
        continents = set()
        for continent in self.main_file['continent']:
            if continent not in continents:
                continents.add(continent)
        radio = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(continents)],
            id='continent_selection',
            switch=True,
            style={
                'overflowY': 'scroll',
                'max-height': '120px'
            }
        )
        return html.Div([
            html.H3('Continent filter'),
            radio,
        ],
            id='continent_filter')

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

    def update_country_options(self, values):
        if values:
            countries_to_show = self.main_file.loc[self.main_file['continent'].isin(values), 'country'].tolist()
        else:
            countries_to_show = self.main_file['country'].tolist()
        return [{'label': i, 'value': i} for i in sorted(set(countries_to_show))]

    def update_place_options(self, values):
        if values:
            places_to_show = self.main_file.loc[self.main_file['country'].isin(values), 'place'].tolist()
        else:
            places_to_show = self.main_file['place'].tolist()
        return [{'label': i, 'value': i} for i in sorted(places_to_show)]

    def update_place_options_by_continent(self, values):
        if values:
            places_to_show = self.main_file.loc[self.main_file['continent'].isin(values), 'place'].tolist()
        else:
            places_to_show = self.main_file['place'].tolist()
        return [{'label': i, 'value': i} for i in sorted(places_to_show)]