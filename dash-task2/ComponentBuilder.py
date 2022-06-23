import nltk
import plotly.express as px
from dash import html, dcc
import plotly.graph_objects as go
nltk.download('stopwords')
import matplotlib.colors as mcolors


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
        df1 = topic_file.loc[topic_file.topic_id==0, :]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df1['word'],
            y=df1['word_count'],
            name='word count',
            marker_color='indianred'
        ))
        fig.add_trace(go.Bar(
            x=df1['word'],
            y=df1['word_count'],
            name='weights',
            width=0.5,
            marker_color='lightsalmon'
        ))

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)

        #fig = px.bar(topic_file, x='word', height='word_count', color='importance')
        #    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        #ax_twin = ax.twinx()
        graph = dcc.Graph(id='topic_charts', figure=fig)

        return html.Div([
            html.H2('Topic charts'),
            graph,
        ])

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
