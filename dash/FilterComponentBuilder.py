import dash_bootstrap_components as dbc
from dash import html


class FilterComponentBuilder:

    def __init__(self, main_file):
        self.main_file = main_file

    def build__filter(self, column_name: int, headline: str):
        """
        Build a new filter composite
        :param column_name: name of column -> will also be prefix of ids
        :param headline: Headline of this filter component
        :return: A div with headline and options
        """
        elements = set(self.main_file[column_name].tolist())
        radio_buttons = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(elements)],
            id=column_name + '_selection',
            switch=True,
            style={
                'overflowY': 'scroll',
                'max-height': '120px'
            }
        )
        html_filter = html.Div([html.H3(headline), radio_buttons], id=column_name + '_filter')
        return html_filter

    def update_options(self, continent_filter, country_filter):
        """
        Update all options and content of site
        :param continent_filter: List of continents to show, empty if show all
        :param country_filter: List of countries to show, empty if show all
        :return: Needed options and content to overwrite current site content
        """
        filtered_countries = self.__extract_filtered_elements(continent_filter, 'continent', 'country')
        if not country_filter:
            country_filter = filtered_countries
        elif filtered_countries:
            matched_filtered_countries = list(set(country_filter).intersection(filtered_countries))
            if matched_filtered_countries:
                country_filter = matched_filtered_countries
            else:
                country_filter = filtered_countries
        filtered_places = self.__extract_filtered_elements(country_filter, 'country', 'place')
        countries = [{'label': i, 'value': i} for i in sorted(set(filtered_countries))]
        places = [{'label': i, 'value': i} for i in sorted(filtered_places)]
        return countries, places, filtered_places

    def __extract_filtered_elements(self, elements, source_column, target_column):
        """
        Filter column elements by condition filter-elements of source column
        :param elements: Values to filter for
        :param source_column: Column to filter elements
        :param target_column: Column to get elements filtered by source column
        :return: Filtered target elements
        """
        if elements:
            return set(self.main_file.loc[
                           self.main_file[source_column].isin(elements), target_column].tolist())
        else:
            return set(self.main_file[target_column].tolist())
