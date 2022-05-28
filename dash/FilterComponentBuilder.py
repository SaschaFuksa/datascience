from ast import literal_eval

import dash_bootstrap_components as dbc
from dash import html


class FilterComponentBuilder:

    @staticmethod
    def build_filter(main_file, column_name: str, headline: str):
        """
        Build a new filter composite
        :param column_name: name of column -> will also be prefix of ids
        :param headline: Headline of this filter component
        :return: A div with headline and options
        """
        colum_data = main_file[column_name].tolist()
        elements = []
        if isinstance(colum_data[0], list):
            for element in colum_data:
                elements.extend(element)
            elements = set(elements)
        elif column_name == 'non_NE_nouns':
            attrs_lists = main_file[column_name].apply(literal_eval)
            for attrs_list in attrs_lists:
                for attraction in attrs_list:
                    elements.append(attraction)
            elements = set(elements)
        else:
            elements = set(main_file[column_name].tolist())
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

    @staticmethod
    def update_options(main_file, continent_filter, country_filter, place_filter):
        """
        Update all options and content of site
        :param continent_filter: List of continents to show, empty if show all
        :param country_filter: List of countries to show, empty if show all
        :return: Needed options and content to overwrite current site content
        """
        filtered_countries = FilterComponentBuilder.extract_filtered_elements(main_file, continent_filter, 'continent',
                                                                              'country')
        if not country_filter:
            country_filter = filtered_countries
        elif filtered_countries:
            matched_filtered_countries = list(set(country_filter).intersection(filtered_countries))
            if matched_filtered_countries:
                country_filter = matched_filtered_countries
            else:
                country_filter = filtered_countries
        filtered_places = FilterComponentBuilder.extract_filtered_elements(main_file, country_filter, 'country',
                                                                           'place')
        if place_filter:
            matched_filtered_places = list(set(filtered_places).intersection(place_filter))
        else:
            matched_filtered_places = filtered_places
        attractions = FilterComponentBuilder.extract_attractions(main_file, matched_filtered_places)
        countries = [{'label': i, 'value': i} for i in sorted(set(filtered_countries))]
        places = [{'label': i, 'value': i} for i in sorted(filtered_places)]
        filtered_attractions = [{'label': i, 'value': i} for i in sorted(attractions)]
        return countries, places, filtered_places, filtered_attractions, attractions

    @staticmethod
    def extract_filtered_elements(main_file, elements, source_column, target_column):
        """
        Filter column elements by condition filter-elements of source column
        :param elements: Values to filter for
        :param source_column: Column to filter elements
        :param target_column: Column to get elements filtered by source column
        :return: Filtered target elements
        """
        if elements:
            return set(main_file.loc[main_file[source_column].isin(elements), target_column].tolist())
        else:
            return set(main_file[target_column].tolist())

    @staticmethod
    def extract_attractions(main_file, matched_filtered_places):
        relevant_attractions = \
            (main_file.loc[main_file['place'].isin(matched_filtered_places), ['non_NE_nouns']])[
                'non_NE_nouns'].apply(literal_eval).tolist()
        attractions_list = []
        for attractions in relevant_attractions:
            attractions_list.extend(attractions)

        return set(attractions_list)
