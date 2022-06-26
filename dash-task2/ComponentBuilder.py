import nltk
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
from sklearn import cluster

nltk.download('stopwords')
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.express as px
from ast import literal_eval


class ComponentBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_clustering(cluster_file):

        df_own = cluster_file.copy()
        #df_own['w2v_label'] = df_own['w2v_label'].astype(str)
        #df_own.rename(columns = {'w2v_label':'Cluster'}, inplace = True)
        df_own['Cluster'] = df_own['w2v_label']
        df_own['Cluster'] = df_own['Cluster'] + 1
        df_own['Cluster'] = df_own['Cluster'].astype(str)

        fig = px.scatter(df_own, x='x', y='y', color="Cluster", hover_data=['place', 'country', 'continent'], template="ggplot2",
            category_orders={'Cluster': ['0','1','2','3','4','5','6','7','8','9']}

        )

        return html.Div([
            html.H2('Clusters'),
            dcc.Graph(id='cluster_dia', figure=fig, config={
                'displayModeBar': False
            }),
        ])

    @staticmethod
    def build_topic_charts(topic_file):
        graph_1 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 0, :], 0, 'indianred',
                                                      'lightsalmon')
        graph_2 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 1, :], 1, 'rgb(102,205,170)',
                                                      'aquamarine')
        graph_3 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 2, :], 2, 'darkkhaki',
                                                      'darkseagreen')
        graph_4 = ComponentBuilder.create_topic_graph(topic_file.loc[topic_file.topic_id == 3, :], 3, 'rgb(0,191,255)', 
                                                    'rgb(0,154,205)')

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
            dcc.Graph(id='top_three_places', style={'width':400}, config={
                'displayModeBar': False
            }),
        ])

    @staticmethod
    def build_top3_countries():
        return html.Div([
            html.H2('Top 3 countries'),
        ])

    @staticmethod
    def build_own_idea(cluster_file):

        df_own = cluster_file.copy()
        #df_own['w2v_label'] = df_own['w2v_label'].astype(str)
        #df_own.rename(columns = {'w2v_label':'Cluster'}, inplace = True)
        df_own['Cluster'] = df_own['w2v_label']
        df_own['Cluster'] = df_own['Cluster'] + 1
        df_own['Cluster'] = df_own['Cluster'].astype(str)

        fig = px.treemap(df_own, 
            path = [px.Constant('All'), 'continent','country', 'place'], 
            values = 'Cluster',
            color = 'Cluster',
            color_continuous_scale = 'GnBu',
            width = 950)
            #height = 450)



        return html.Div([
            html.H2('Own idea'),
            dcc.Graph(id='own_dia', figure=fig, style={'colspan': 2}, config={
                'displayModeBar': False
            })
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

        colours = ['102,205,170', '58, 71, 80', '71, 58, 131', '60, 163, 204']
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
                    color='rgba(' + colour + ', 0.6)',
                    line=dict(color='rgba(' + colour + ', 1.0)', width=1.5)
                ),
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
        for row in top3_places_df.itertuples():
            place = row.place
            attractions = literal_eval(row.singular_cleaned_nouns)
            freqs = literal_eval(row.freq_noun_int)
            list_of_freqs = [0, 0, 0, 0]
            list_of_freqs = list_of_freqs[:len(filter_list)]
            for attr, freq in zip(attractions, freqs):
                if attr in filter_list:
                    index = filter_list.index(attr)
                    list_of_freqs[index] = freq
            new_col = [place]
            new_col.extend(list_of_freqs)
            df.loc[len(df.index)] = new_col

        df_2 = df[~(df == 0).any(axis=1)]
        if len(df_2) >= len(filter_list):
            df = df_2
        df['sum'] = df.sum(axis=1)
        df = df.sort_values(by=['sum'], ascending=False)
        df = df[:3]
        return df

    @staticmethod
    def update_cluser(cluster_file, cluster_filter):

        cluster_file['w2v_label'] = cluster_file['w2v_label'].astype(str)
        #cluster_file.rename(columns = {'w2v_label':'Cluster'}, inplace = True)
        cluster_file['Cluster'] = cluster_file['w2v_label']

        fig = px.scatter(cluster_file, x='x', y='y', color="Cluster", hover_data=['place', 'country', 'continent'], template="ggplot2",
            category_orders={'Cluster': ['0','1','2','3','4','5','6','7','8','9']}

        )

        return fig