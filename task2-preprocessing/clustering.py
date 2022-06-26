#%%
#imports
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import gensim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.cluster import hierarchy

#num clusters
clusterNum = 10 

#%%
#load data from task1
df_places = pd.read_csv('../crawling-and-preprocessing/content/data_prep_2805_3.csv', usecols=['place', 'country', 'continent', 'singular_nouns'])
df_places['singular_nouns'] = df_places['singular_nouns'].apply(literal_eval)
df_places

#convert string array to one long string
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
#cluster with K-Means based on word count

model = KMeans(n_clusters = clusterNum)
model.fit(count_matrix)

#%%
#save label and cluster into variables
labels = model.labels_
cluster_center=model.cluster_centers_

#%%
#check silhoute score
silhouette_score = metrics.silhouette_score(count_matrix, labels, metric='euclidean')
silhouette_score

#%%
#append new column with cluster-labels
df_places['count_label'] = model.labels_ # the last column you can see the label numebers
df_places.groupby(['count_label'])['cleaned_text'].count()

#%%
#show top terms per cluster
#used to check if cluster results are acceptable
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

#%%
#reduce the features to 2D
#allows for visualisazion with scatter-plot

pca = TruncatedSVD(n_components=2, random_state=42)
reduced_features = pca.fit_transform(count_matrix)

#reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(model.cluster_centers_)

#plot 2D scatter
plt.scatter(reduced_features[:,0], reduced_features[:,1], model.predict(count_matrix))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

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
#cluster with k-means based on word count

model_tf = KMeans(n_clusters = clusterNum)
model_tf.fit(tfidf_matrix)

#%%
#save labels and cluster center into variables
labels_tf = model_tf.labels_
cluster_center_tf=model_tf.cluster_centers_

#%%
#check silhouette score
silhouette_score_tf = metrics.silhouette_score(tfidf_matrix, labels_tf, metric='euclidean')
silhouette_score_tf

#%%
#append new column with result labels
df_places['tfidf_label'] = model_tf.labels_ # the last column you can see the label numebers
df_places.groupby(['tfidf_label'])['cleaned_text'].count()

#%%
#show top terms per cluster
#used to check if cluster results are acceptable
print("Top terms per cluster:")
order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]
terms_tfidf = tfidf_vect.get_feature_names()
for i in range(10):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms_tfidf[ind], end='')
        print()

#%%
#plot distribution of places into cluster
plt.bar([x for x in range(10)], df_places.groupby(['tfidf_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
# reduce the features to 2D
pca = TruncatedSVD(n_components=2, random_state=42)
reduced_features = pca.fit_transform(tfidf_matrix)

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(model_tf.cluster_centers_)

#plot 2d scatter
plt.scatter(reduced_features[:,0], reduced_features[:,1], model_tf.predict(tfidf_matrix))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

#%%
df_places
####################################################
#   K-Means with Average word to vector            #
####################################################
#%%
#create list of token
i=0
list_of_w2v=[]
for sent in df_places['cleaned_text'].values:
    list_of_w2v.append(sent.split())

#%%
#train model with token/words
w2v_model=gensim.models.Word2Vec(list_of_w2v, workers=4)



#%%
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
#Choosing the best cluster using Elbow Method.
#number of clusters to check
#cluster each number of clusters and detect best number of cells
num_clus = [x for x in range(3,20)]
squared_errors = []
for cluster in num_clus:
    kmeans = KMeans(n_clusters = cluster).fit(sent_vectors)
    squared_errors.append(kmeans.inertia_)

optimal_clusters = np.argmin(squared_errors) + 2
plt.plot(num_clus, squared_errors)
xy = (optimal_clusters, min(squared_errors))
plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
plt.show()


#%%
#training with optimal cluster
#using the created gensim model to cluster places
model_w2v = KMeans(n_clusters = clusterNum)
model_w2v.fit(sent_vectors)

word_cluster_pred=model_w2v.predict(sent_vectors)
word_cluster_pred_2=model_w2v.labels_
word_cluster_center=model_w2v.cluster_centers_

#append cluster labels to data
df_places['w2v_label'] = model_w2v.labels_
df_places.groupby(['w2v_label'])['cleaned_text'].count()

#plot the calculated clusters
plt.bar([x for x in range(clusterNum)], df_places.groupby(['w2v_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
# reduce the features to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
reduced_features = pca.fit_transform(sent_vectors)

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(model_w2v.cluster_centers_)

#plot the reduced dimensions
plt.scatter(reduced_features[:,0], reduced_features[:,1], model_w2v.predict(sent_vectors))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


#%%
#append 2d coordinates to data
#will be used to visualize the clustering
df_places['x'] = reduced_features[:,0]
df_places['y'] = reduced_features[:,1]
df_places

####################################################
#   Clustering with DBSCAN                         #
####################################################
#%%
#density based clustering with dbscan
#calculate nearest neighbours
nearest_neighbors = NearestNeighbors(n_neighbors=clusterNum)
neighbors = nearest_neighbors.fit(sent_vectors)

distances, indices = neighbors.kneighbors(sent_vectors)
distances = np.sort(distances[:,clusterNum-1], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")

#%%
#knee method to determine best eps
#optimal point is in the middle of the "knee"
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])

#%%
#use determined eps to cluster data
dbscan_cluster = DBSCAN(eps=0.061)
dbscan_cluster.fit(sent_vectors)

#get number of clusters
labels=dbscan_cluster.labels_
N_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % N_clus)

#%%
#append new labels with cluster
df_places['dbscan_label'] = dbscan_cluster.labels_
df_places.groupby(['dbscan_label'])['cleaned_text'].count()

#%%
#plot distribution of labels
plt.bar([x for x in range(2)], df_places.groupby(['dbscan_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
# reduce the features to 2D and plot
pca = PCA(n_components=2, random_state=42)
reduced_features = pca.fit_transform(sent_vectors)
plt.scatter(reduced_features[:,0], reduced_features[:,1], dbscan_cluster.fit(sent_vectors))

####################################################
#   Clustering hierarchical                        #
####################################################
#%%
dendro=hierarchy.dendrogram(hierarchy.linkage(sent_vectors,method='ward'))
plt.axhline(y=35)

#%%
#cluster hierarchically and apend labels to data
cluster = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='ward')  #took n=5 from dendrogram curve 
Agg=cluster.fit_predict(sent_vectors)
df_places['h_label'] = cluster.labels_
df_places.groupby(['h_label'])['cleaned_text'].count()

#%%
#plot the data
plt.bar([x for x in range(19)], df_places.groupby(['h_label'])['cleaned_text'].count(), alpha = 0.4)
plt.title('KMeans cluster points')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#%%
# reduce the features to 2D and plot
pca = PCA(n_components=2, random_state=42)
reduced_features = pca.fit_transform(sent_vectors)
plt.scatter(reduced_features[:,0], reduced_features[:,1], cluster.fit_predict(sent_vectors))

#%%
df_places.to_csv('location_cluster.csv', index=False)


#%%

df_temp = pd.read_csv('../crawling-and-preprocessing/content/location_cluster.csv')
df_temp

# %%
