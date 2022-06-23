import nltk
import plotly.graph_objects as go
from dash import html, dcc

nltk.download('stopwords')
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc


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
        fig = make_subplots(specs=[[{"secondary_y": True}]])

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

        fig.update_layout(barmode='group', xaxis_tickangle=-45, margin=dict(l=0, r=0, t=0, b=0),)
        graph = dcc.Graph(id='topic_charts_' + str(id), figure=fig, config={
            'displayModeBar': False
        })
        return graph

    @staticmethod
    def build_top3_places():
        return html.Div([
            html.H2('Top 3 places'),
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
