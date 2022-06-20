#%%
import pandas as pd
import umap
import umap.plot
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


#%%
df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'non_NE_nouns'])
#df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['non_NE_tags'])
df_places['non_NE_nouns'] = df_places['non_NE_nouns'].apply(literal_eval)
#df_places['non_NE_nouns'] = df_places['non_NE_tags'].apply(literal_eval)
df_places['non_NE_nouns']

#%%
temp_noun = []
temp_hyper = []

for row in df_places.itertuples():
    temp_n = []
    temp_h = []
    for t in row.non_NE_tags:
        temp_n.append(t[0])
        if t[1] != '-':
            temp_h.append(t[1])

    temp_noun.append(temp_n)
    temp_hyper.append(temp_h)

df_temp = pd.DataFrame({'noun':temp_noun,'hyper':temp_hyper})
df_temp

df_temp['hyper'] = [' '.join(map(str, l)) for l in df_temp['hyper']]
df_temp['hyper'] = df_temp['hyper'].map(lambda x: x.lower())
df_temp

#%%
df_temp = pd.DataFrame(columns=['noun','hyper'])
df_temp[['noun', 'hyper']] = pd.DataFrame(df_places['non_NE_tags'].tolist(), index=df_places.index)
df_temp

#%%
df_places['non_NE_tags'][1][1][0]

#%%
type(df_places['non_NE_tags'][1][1])

# %%
for row in df_places.itertuples():
    print(type(row.non_NE_tags))
    print(row.non_NE_tags)
    break

#%%
df_places['non_NE_nouns'] = [' '.join(map(str, l)) for l in df_places['non_NE_nouns']]
df_places['non_NE_nouns'] = df_places['non_NE_nouns'].map(lambda x: x.lower())
df_places

#%%
print(df_places['non_NE_nouns'][1].lower())

# %%
tfid_vectorizer = TfidfVectorizer(stop_words='english')
#tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['non_NE_nouns'])
tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['non_NE_nouns'])
tfid_vectorizer.get_feature_names_out()

#%%
tfid_word_doc_matrix.shape

#%%
from sklearn.cluster import KMeans
true_k = 48
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(tfid_word_doc_matrix)
labels=model.labels_
df_places['label'] = labels
df_places

#%%
df_temp['label'].max()

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
import numpy as np
def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(tfid_word_doc_matrix.todense()).groupby(labels).mean() # groups the TF-IDF vector by cluster
    terms = tfid_vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            
get_top_keywords(10)
#%%
# viz libs
import matplotlib.pyplot as plt
import seaborn as sns
# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("Hypernymen Clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df_places, x='x0', y='x1', hue='label', palette="viridis")
plt.show()

#%%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Sum_of_squared_distances = []
K = range(1,100)

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
long_string = ','.join(list(df_places['non_NE_nouns'].values))

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
word_doc_matrix = vectorizer.fit_transform(df_places['non_NE_nouns'])

#%%
reducer = umap.UMAP(n_components=2, metric='cosine')
embedding = reducer.fit(word_doc_matrix)

#%%
f = umap.plot.points(embedding, labels=df_places['place'])
# %%
tfid_vectorizer = TfidfVectorizer(stop_words='english')
tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['non_NE_nouns'])

#%%
embedding2 = reducer.fit(tfid_word_doc_matrix)
p = umap.plot.points(embedding2, labels=df_places['place'])
# %%
for x in tfid_word_doc_matrix:
    print(x)
# %%
