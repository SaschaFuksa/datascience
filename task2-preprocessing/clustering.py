#%%
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import gensim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

clusterNum = 10

#%%
#load data from task1
df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'singular_nouns'])
df_places['singular_nouns'] = df_places['singular_nouns'].apply(literal_eval)
df_places['singular_nouns']

#%%
#cleaning string-array and create new column
df_places['cleaned_text'] = [' '.join(map(str, l)) for l in df_places['singular_nouns']]
df_places['cleaned_text'] = df_places['cleaned_text'].map(lambda x: x.lower())
df_places.drop(columns='singular_nouns', inplace=True)
df_places


####################################################
#   K-Means with bag of words                      #
####################################################
#%%
#Create matrix with counted features
count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(df_places['cleaned_text'].values)
count_terms = count_vect.get_feature_names()
count_terms[1:10]

#%%
model = KMeans(n_clusters = clusterNum,init='k-means++',random_state=99)
model.fit(count_matrix)

#%%
labels = model.labels_
cluster_center=model.cluster_centers_

#%%
silhouette_score = metrics.silhouette_score(count_matrix, labels, metric='euclidean')
silhouette_score

#%%
df_places['count_label'] = model.labels_ # the last column you can see the label numebers
df_places.groupby(['count_label'])['cleaned_text'].count()

#%%
#show top terms per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms_count = count_vect.get_feature_names()
for i in range(10):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms_count[ind], end='')
        print()

#%%
#visualization of distribution
plt.bar([x for x in range(10)], df_places.groupby(['count_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()


####################################################
#   K-Means with TFIDF                             #
####################################################
#%%
#create tfidf matrix
tfidf_vect = TfidfVectorizer()
tfidf_matrix = tfidf_vect.fit_transform(df_places['cleaned_text'].values)
tfid_terms = tfidf_vect.get_feature_names()
tfid_terms[1:10]

#%%
model_tf = KMeans(n_clusters = clusterNum,random_state=99)
model_tf.fit(tfidf_matrix)

#%%
labels_tf = model_tf.labels_
cluster_center_tf=model_tf.cluster_centers_

#%%
silhouette_score_tf = metrics.silhouette_score(tfidf_matrix, labels_tf, metric='euclidean')
silhouette_score_tf

#%%
df_places['tfidf_label'] = model_tf.labels_ # the last column you can see the label numebers
df_places.groupby(['tfidf_label'])['cleaned_text'].count()

#%%
print("Top terms per cluster:")
order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]
terms_tfidf = tfidf_vect.get_feature_names()
for i in range(10):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms_tfidf[ind], end='')
        print()

#%%
plt.bar([x for x in range(10)], df_places.groupby(['tfidf_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

####################################################
#   K-Means with Average word to vector            #
####################################################
#%%
i=0
list_of_w2v=[]
for sent in df_places['cleaned_text'].values:
    list_of_w2v.append(sent.split())
print(df_places['cleaned_text'].values[0])
print("*****************************************************************")
print(list_of_w2v[0])

#%%
w2v_model=gensim.models.Word2Vec(list_of_w2v, workers=4)

#%%
import numpy as np
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this train
for sent in list_of_w2v: # for each review/sentence
    sent_vec = np.zeros(100) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
sent_vectors = np.array(sent_vectors)
sent_vectors = np.nan_to_num(sent_vectors)
sent_vectors.shape

#%%
# Number of clusters to check.
num_clus = [x for x in range(3,20)]


#%%
# Choosing the best cluster using Elbow Method.
squared_errors = []
for cluster in num_clus:
    kmeans = KMeans(n_clusters = cluster).fit(sent_vectors) # Train Cluster
    squared_errors.append(kmeans.inertia_) # Appending the squared loss obtained in the list

optimal_clusters = np.argmin(squared_errors) + 2 # As argmin return the index of minimum loss.
plt.plot(num_clus, squared_errors)
plt.title("Elbow Curve to find the no. of clusters.")
plt.xlabel("Number of clusters.")
plt.ylabel("Squared Loss.")
xy = (optimal_clusters, min(squared_errors))
plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
plt.show()


#%%
#training with optimal cluster
#model_w2v = KMeans(n_clusters = optimal_clusters)
temp_cluster = 10
model_w2v = KMeans(n_clusters = temp_cluster)
model_w2v.fit(sent_vectors)

word_cluster_pred=model_w2v.predict(sent_vectors)
word_cluster_pred_2=model_w2v.labels_
word_cluster_center=model_w2v.cluster_centers_

df_places['w2v_label'] = model_w2v.labels_
df_places.groupby(['w2v_label'])['cleaned_text'].count()

plt.bar([x for x in range(temp_cluster)], df_places.groupby(['w2v_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
df_places.groupby(['w2v_label'])['cleaned_text'].count()
#df_places.sort_values(by='w2v_label', ascending=False)

####################################################
#   Clustering with DBSCAN                         #
####################################################
#%%
nearest_neighbors = NearestNeighbors(n_neighbors=10)
neighbors = nearest_neighbors.fit(sent_vectors)

distances, indices = neighbors.kneighbors(sent_vectors)
distances = np.sort(distances[:,9], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")

#%%
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])

#%%
dbscan_cluster = DBSCAN(eps=0.061)
dbscan_cluster.fit(sent_vectors)

# Number of Clusters
labels=dbscan_cluster.labels_
N_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % N_clus)

# Identify Noise
n_noise = list(dbscan_cluster.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)

# Calculating v_measure
#print('v_measure =', metrics.v_measure_score(y, labels))

#%%
df_places['dbscan_label'] = dbscan_cluster.labels_
df_places.groupby(['dbscan_label'])['cleaned_text'].count()

#%%
df_places['dbscan_label']

#%%
plt.bar([x for x in range(2)], df_places.groupby(['dbscan_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

####################################################
#   Clustering hierarchical                        #
####################################################
#%%
import scipy
from scipy.cluster import hierarchy
dendro=hierarchy.dendrogram(hierarchy.linkage(sent_vectors,method='ward'))
plt.axhline(y=35)# cut at 30 to get 5 clusters
# %%
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='ward')  #took n=5 from dendrogram curve 
Agg=cluster.fit_predict(sent_vectors)
# %%
df_places['h_label'] = cluster.labels_
df_places.groupby(['h_label'])['cleaned_text'].count()
# %%
#%%
plt.bar([x for x in range(10)], df_places.groupby(['h_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()
# %%
df_places
# %%
