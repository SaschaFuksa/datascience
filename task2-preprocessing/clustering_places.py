#%%
import pandas as pd
import umap.umap_ as umap
import umap.plot
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'singular_cleaned_nouns'])
df_places['singular_cleaned_nouns'] = df_places['singular_cleaned_nouns'].apply(literal_eval)
df_places
# %%
for row in df_places.itertuples():
    print(type(row.singular_cleaned_nouns))
    print(row.singular_cleaned_nouns)
    break

#%%
df_places['nouns_string'] = [' '.join(map(str, l)) for l in df_places['singular_cleaned_nouns']]
df_places
# %%
vectorizer = CountVectorizer()
word_doc_matrix = vectorizer.fit_transform(df_places['nouns_string'])

#%%
embedding = umap(n_components=2, metric='cosine').fit(word_doc_matrix)

#%%
