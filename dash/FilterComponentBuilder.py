import dash_bootstrap_components as dbc
from dash import html


class FilterComponentBuilder:

    def __init__(self):
        pass

    def build__filter(df, column_name: int, headline: str):
        """
        Build a new filter composite
        :param column_name: name of column -> will also be prefix of ids
        :param headline: Headline of this filter component
        :return: A div with headline and options
        """
        colum_data = df[column_name].tolist()
        elements = []
        if isinstance(colum_data[0], list):
            for element in colum_data:
                elements.extend(element)
            elements = set(elements)
        else:
            elements = set(df[column_name].tolist())
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

    def update_options(df, example_attractions, continent_filter, country_filter, place_filter):
        """
        Update all options and content of site
        :param continent_filter: List of continents to show, empty if show all
        :param country_filter: List of countries to show, empty if show all
        :return: Needed options and content to overwrite current site content
        """
        filtered_countries = FilterComponentBuilder.extract_filtered_elements(df, continent_filter, 'continent',
                                                                              'country')
        if not country_filter:
            country_filter = filtered_countries
        elif filtered_countries:
            matched_filtered_countries = list(set(country_filter).intersection(filtered_countries))
            if matched_filtered_countries:
                country_filter = matched_filtered_countries
            else:
                country_filter = filtered_countries
        filtered_places = FilterComponentBuilder.extract_filtered_elements(df, country_filter, 'country', 'place')
        if place_filter:
            matched_filtered_places = list(set(filtered_places).intersection(place_filter))
        else:
            matched_filtered_places = filtered_places
        attractions = FilterComponentBuilder.extract_attractions(example_attractions, matched_filtered_places)
        countries = [{'label': i, 'value': i} for i in sorted(set(filtered_countries))]
        places = [{'label': i, 'value': i} for i in sorted(filtered_places)]
        filtered_attractions = [{'label': i, 'value': i} for i in sorted(attractions)]
        return countries, places, filtered_places, filtered_attractions

    def extract_filtered_elements(df, elements, source_column, target_column):
        """
        Filter column elements by condition filter-elements of source column
        :param elements: Values to filter for
        :param source_column: Column to filter elements
        :param target_column: Column to get elements filtered by source column
        :return: Filtered target elements
        """
        if elements:
            return set(df.loc[df[source_column].isin(elements), target_column].tolist())
        else:
            return set(df[target_column].tolist())

    def extract_attractions(example_attractions, matched_filtered_places):
        relevant_attractions = \
            (example_attractions.loc[example_attractions['place'].isin(matched_filtered_places), ['word']])[
                'word'].tolist()
        attractions_list = []
        for attractions in relevant_attractions:
            attractions_list.extend(attractions)
        return set(attractions_list)
