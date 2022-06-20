#%%
import pandas as pd
import umap
import umap.plot
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#%%
df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'singular_nouns'])
#df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['non_NE_tags'])
df_places['singular_nouns'] = df_places['singular_nouns'].apply(literal_eval)
#df_places['non_NE_nouns'] = df_places['non_NE_tags'].apply(literal_eval)
df_places['singular_nouns']

#%%
df_places['cleaned_text'] = [' '.join(map(str, l)) for l in df_places['singular_nouns']]
df_places['cleaned_text'] = df_places['cleaned_text'].map(lambda x: x.lower())
df_places

#%%
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
bow = count_vect.fit_transform(df_places['cleaned_text'].values)
bow.shape

#%%
terms = count_vect.get_feature_names()
terms[1:10]

#%%
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 10,init='k-means++',random_state=99)
model.fit(bow)

#%%
labels = model.labels_
cluster_center=model.cluster_centers_

#%%
from sklearn import metrics
silhouette_score = metrics.silhouette_score(bow, labels, metric='euclidean')
silhouette_score

#%%
# Giving Labels/assigning a cluster to each point/text 
#df_places = final
df_places['label'] = model.labels_ # the last column you can see the label numebers
df_places.head(20)

#%%
# How many points belong to each cluster -> using group by in pandas
df_places.groupby(['label'])['singular_nouns'].count()

#%%
#Refrence credit - to find the top 10 features of cluster centriod
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = count_vect.get_feature_names()
for i in range(10):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        print()

#%%
# visually how points or reviews are distributed across 10 clusters 
import matplotlib.pyplot as plt
plt.bar([x for x in range(10)], df_places.groupby(['label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
#tfidf vector initililization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(df_places['cleaned_text'].values)
tfidf.shape

#%%
from sklearn.cluster import KMeans
model_tf = KMeans(n_clusters = 10, random_state=99)
model_tf.fit(tfidf)

#%%
labels_tf = model_tf.labels_
cluster_center_tf=model_tf.cluster_centers_

#%%
from sklearn import metrics
silhouette_score_tf = metrics.silhouette_score(tfidf, labels_tf, metric='euclidean')
silhouette_score_tf

#%%
df_places['tf_label'] = model_tf.labels_
df_places.head(5)

#%%
df_places.groupby(['tf_label'])['cleaned_text'].count()

#%%
#Refrence credit - to find the top 10 features of cluster centriod
print("Top terms per cluster:")
order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]
for i in range(10):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        print()

#%%
# visually how points or reviews are distributed across 10 clusters 
plt.bar([x for x in range(10)], df_places.groupby(['tf_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()





#----------------------------------------------------#
# OLD
#%%
tfid_vectorizer = TfidfVectorizer(stop_words='english')
#tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['non_NE_nouns'])
tfid_word_doc_matrix = tfid_vectorizer.fit_transform(df_places['non_NE_nouns'])
tfid_vectorizer.get_feature_names_out()

#%%
from sklearn.cluster import AgglomerativeClustering

X = tfid_word_doc_matrix.toarray()
cluster = AgglomerativeClustering(n_clusters=48, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

print(cluster.labels_)
# %%
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')

# %%
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(X, method='ward'))


# %%
plt.figure(figsize=(10, 7))
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
# %%
