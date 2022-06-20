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

# Average Word to Vector
#%%
i=0
list_of_sent=[]
for sent in df_places['cleaned_text'].values:
    list_of_sent.append(sent.split())
print(df_places['cleaned_text'].values[0])
print("*****************************************************************")
print(list_of_sent[0])

#%%
# removing html tags and apostrophes if present.
import re
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

#%%
i=0
list_of_sent_train=[]
for sent in df_places['cleaned_text'].values:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent_train.append(filtered_sentence)

#%%
import gensim
# Training the wor2vec model using train dataset
w2v_model=gensim.models.Word2Vec(list_of_sent_train, workers=4)

#%%
import numpy as np
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this train
for sent in list_of_sent_train: # for each review/sentence
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
num_clus = [x for x in range(3,11)]
num_clus

#%%
# Choosing the best cluster using Elbow Method.
# source credit,few parts of min squred loss info is taken from different parts of the stakoverflow answers.
# this is used to understand to find the optimal clusters in differen way rather than used in BOW, TFIDF
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

print ("The optimal number of clusters obtained is - ", optimal_clusters)
print ("The loss for optimal cluster is - ", min(squared_errors))

#%%
# Training the best model --
from sklearn.cluster import KMeans
model2 = KMeans(n_clusters = optimal_clusters)
model2.fit(sent_vectors)

#%%
word_cluster_pred=model2.predict(sent_vectors)
word_cluster_pred_2=model2.labels_
word_cluster_center=model2.cluster_centers_
word_cluster_center[1:2]

#%%
# Giving Labels/assigning a cluster to each point/text 
df_places['avg_label'] = model2.labels_
df_places.head(2)

#%%
df_places.groupby(['avg_label'])['cleaned_text'].count()

#%%
# visually how points or reviews are distributed across 10 clusters 
plt.bar([x for x in range(9)], df_places.groupby(['avg_label'])['cleaned_text'].count(), alpha = 0.4)
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
