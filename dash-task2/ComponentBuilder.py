import nltk
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc

nltk.download('stopwords')
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.express as px
from ast import literal_eval


class ComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_scatter():
        return html.Div([
            html.H2('Attraction clusters'),
        ])

    @staticmethod
    def build_topic_charts(topic_file):
        graph_1 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 0, :], 0, 'indianred',
                                                      'lightsalmon')
        graph_2 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 1, :], 1, 'aqua',
                                                      'aquamarine')
        graph_3 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 2, :], 2, 'darkkhaki',
                                                      'darkseagreen')
        graph_4 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 3, :], 3, 'gold',
                                                      'goldenrod')

        return html.Div([
            html.H2('Topic charts'),
            html.Div(children=[
                dbc.Row([
                    dbc.Col(graph_1),
                    dbc.Col(graph_2)
                ]),
                dbc.Row([
                    dbc.Col(graph_3),
                    dbc.Col(graph_4)
                ]),
            ])
        ])

    @staticmethod
    def create_topic_graph(df, id, colour_1, colour_2):
        fig = make_subplots(specs=[[{'secondary_y': True}]])

        fig.add_trace(go.Bar(
            x=df['word'],
            y=df['word_count'],
            name='word count',
            marker_color=colour_1,
        ), secondary_y=False, )
        fig.add_trace(go.Bar(
            x=df['word'],
            y=df['importance'],
            name='weights',
            width=0.3,
            marker_color=colour_2
        ), secondary_y=True, )

        fig.update_layout(barmode='group', xaxis_tickangle=-45, margin=dict(l=0, r=0, t=0, b=0), height=200,
                          legend=dict(
                              orientation='h',
                              yanchor='bottom',
                              y=1.02,
                              xanchor='right',
                              x=1
                          ))
        graph = dcc.Graph(id='topic_charts_' + str(id), figure=fig, config={
            'displayModeBar': False
        })

        return graph

    @staticmethod
    def build_top3_places():
        return html.Div([
            html.H2('Top 3 places'),
            dcc.Graph(id='top_three_places', config={
            'displayModeBar': False
        }),
        ])

    @staticmethod
    def build_top3_countries():
        return html.Div([
            html.H2('Top 3 countries'),
        ])

    @staticmethod
    def build_own_idea():
        return html.Div([
            html.H2('Own idea'),
        ])

    @staticmethod
    def update_top3_places(main_file, filters):
        top3_places_df = main_file[['place', 'singular_cleaned_nouns', 'freq_noun_int']]
        new_filter = []
        for filter in filters:
            if filter:
                new_filter.append(filter)

        if all(x is None for x in filters):
            data, filters = ComponentBuilder.search_in_all(top3_places_df)
        else:
            filters = new_filter
            data = ComponentBuilder.search_in_filter(top3_places_df, filters)

        colours = ['246, 78, 139', '58, 71, 80', '71, 58, 131', '164, 163, 204']
        max_len = len(filters)
        colours = colours[:max_len]
        fig = go.Figure()
        for filter, colour in zip(filters, colours):
            fig.add_trace(go.Bar(
                y=data['place'],
                x=data[filter],
                name=filter,
                orientation='h',
                marker=dict(
                    color='rgba('+colour+', 0.6)',
                    line=dict(color='rgba('+colour+', 1.0)', width=3)
                )
            ))
        fig.update_layout(barmode='stack', xaxis_tickangle=-45, margin=dict(l=0, r=0, t=0, b=0), height=200,
                          legend=dict(
                              orientation='h',
                              yanchor='bottom',
                              y=1.02,
                              xanchor='right',
                              x=1
                          ))

        return fig

    @staticmethod
    def search_in_all(top3_places_df):
        attr_list = []
        freq_list = []
        for nes in top3_places_df['singular_cleaned_nouns'].apply(literal_eval):
            attr_list.extend(nes)
        for freq in top3_places_df['freq_noun_int'].apply(literal_eval):
            freq_list.extend(map(int, freq))

        most_distributed_attributes = pd.DataFrame({'singular_cleaned_nouns': attr_list, 'freq_noun_int': freq_list})
        most_distributed_attributes = most_distributed_attributes.sort_values(by=['freq_noun_int'], ascending=False)
        most_distributed_attributes = most_distributed_attributes.drop_duplicates(subset='singular_cleaned_nouns',
                                                                                  keep="first")
        most_distributed_attributes = most_distributed_attributes[:4]

        filter_list = list(most_distributed_attributes['singular_cleaned_nouns'])
        return ComponentBuilder.search_in_filter(top3_places_df, filter_list), filter_list

    @staticmethod
    def search_in_filter(top3_places_df, filter_list):
        list_cols = ['place']
        list_cols.extend(filter_list)
        df = pd.DataFrame(columns=list_cols)
        # Hier generisch um je nach col Anzahl durchzugehen
        for row in top3_places_df.itertuples():
            place = row.place
            attractions = literal_eval(row.singular_cleaned_nouns)
            freqs = literal_eval(row.freq_noun_int)
            freq_0 = 0
            freq_1 = 0
            freq_2 = 0
            freq_3 = 0
            for attr, freq in zip(attractions, freqs):
                if attr in filter_list:
                    index = filter_list.index(attr)
                    if index == 0:
                        freq_0 = freq
                    if index == 1:
                        freq_1 = freq
                    if index == 2:
                        freq_2 = freq
                    if index == 3:
                        freq_3 = freq
            df = df.append({'place': place, filter_list[0]: freq_0, filter_list[1]: freq_1, filter_list[2]: freq_2,
                            filter_list[3]: freq_3}, ignore_index=True)
        df = df[(df[filter_list[0]] != 0) & (df[filter_list[1]] != 0) & (df[filter_list[2]] != 0) & (
                    df[filter_list[3]] != 0)]
        df['sum'] = df[filter_list[0]] + df[filter_list[1]] + df[filter_list[2]] + df[filter_list[3]]
        df = df.sort_values(by=['sum'], ascending=False)
        df = df[:3]
        return df