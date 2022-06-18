#%%
import pandas as pd
import umap
import umap.plot
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'singular_cleaned_nouns'])
df_places['singular_cleaned_nouns'] = df_places['singular_cleaned_nouns'].apply(literal_eval)
#df_places['nouns_string'] = df_places['singular_cleaned_nouns'].apply(literal_eval)
df_places
# %%
for row in df_places.itertuples():
    print(type(row.singular_cleaned_nouns))
    print(row.singular_cleaned_nouns)
    break

#%%
df_places['nouns_string'] = [' '.join(map(str, l)) for l in df_places['singular_cleaned_nouns']]
df_places['nouns_string'] = df_places['nouns_string'].map(lambda x: x.lower())
df_places

#%%
print(df_places['nouns_string'][1].lower())

# %%
tfid_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['nouns_string'])
tfid_vectorizer.get_feature_names_out()

#%%
tfid_word_doc_matrix.shape

#%%
true_k = 48
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(tfid_word_doc_matrix)
labels=model.labels_
df_places['label'] = labels

#%%
from sklearn.decomposition import PCA

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(tfid_word_doc_matrix.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]
df_places['x0'] = x0
df_places['x1'] = x1



#%%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Sum_of_squared_distances = []
K = range(47,50)

for k in K:
   km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   km = km.fit(tfid_word_doc_matrix)
   Sum_of_squared_distances.append(km.inertia_)
   
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


#%%
df_places.sort_values(by=['label'])



#%%
# Import the wordcloud library
from wordcloud import WordCloud

# Join the different processed titles together.
long_string = ','.join(list(df_places['nouns_string'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()















#---------------------------------------#
# OLD
#---------------------------------------#


# %%
vectorizer = CountVectorizer()
word_doc_matrix = vectorizer.fit_transform(df_places['nouns_string'])

#%%
reducer = umap.UMAP(n_components=2, metric='cosine')
embedding = reducer.fit(word_doc_matrix)

#%%
f = umap.plot.points(embedding, labels=df_places['place'])
# %%
tfid_vectorizer = TfidfVectorizer(stop_words='english')
tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['nouns_string'])

#%%
embedding2 = reducer.fit(tfid_word_doc_matrix)
p = umap.plot.points(embedding2, labels=df_places['place'])
# %%
for x in tfid_word_doc_matrix:
    print(x)
# %%
