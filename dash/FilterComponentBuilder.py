import dash_bootstrap_components as dbc


class FilterComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_country_filter(country_column):
        countries = set()
        countries.add('*')
        for country in country_column:
            if country not in countries:
                countries.add(country)
        radio = dbc.RadioItems(
            options=[{'label': i, 'value': i} for i in sorted(countries)],
            id='country_selection',
            value='*',
        )
        return radio

    @staticmethod
    def build_place_filter(place_column):
        places = []
        for place in place_column:
            if place not in places:
                places.append(place)
        check_list = dbc.Checklist(
            options=[{'label': i, 'value': i} for i in sorted(places)],
            id='place_selection',
            value=places
        )
        return check_list
