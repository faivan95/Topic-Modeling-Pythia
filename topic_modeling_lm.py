# import the libraries 
from transformers import GPTNeoXForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.cm as cm

# from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import umap
from sklearn.datasets import fetch_20newsgroups
# import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from soyclustering import SphericalKMeans
from sklearn.cluster import SpectralClustering
#from spherecluster import SphericalKMeans
from scipy.sparse import csr_matrix


def sort_closest_center(center, cluster_ids, word_embeddings, c_idx):
    #return the index of words in cluster i
    data_idx_within_i_cluster = np.array([idx for idx, cluster_id in enumerate(cluster_ids) if cluster_id == c_idx])
    each_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), center.shape[0])) 

    # Get the word embedding of the same cluster
    for row_num, data_idx in enumerate(data_idx_within_i_cluster):
        each_cluster_tf_matrix[row_num] = word_embeddings[data_idx]

    # Create the distance of each word embeddings of a cluster to its center
    dist_X = np.sum((each_cluster_tf_matrix - center)**2, axis=1)

    # Return the index of sorted top words
    topk_values_idx = dist_X.argsort().astype(int)
    topk_values = data_idx_within_i_cluster[topk_values_idx]

    return topk_values

def find_top_k_words(k, top_values, corpus):

    ind = []
    # Create unique words set
    unique_words = set()

    for i in top_values:
        word = tokenizer.decode(corpus[i])
        word = word.replace(" ", "")
        if word not in unique_words and len(word) > 2 and word != "[PAD]" and word != "\n" and word != "\t":
            ind.append(word)
            unique_words.add(word)
            if len(unique_words) == k:
                break
    return ind

def get_clusters_kmeans(word_embeddings, num_clusters, topk, corpus, weights):
    # Fit the kmeans model to the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=20, init='k-means++', n_init=10, max_iter=500).fit(word_embeddings, sample_weight=weights)
    # Return the labels of words and center of each cluster
    # clusters = kmeans.labels_
    clusters = kmeans.predict(word_embeddings, sample_weight=weights)
    centers = np.array(kmeans.cluster_centers_)

    indices = []

    for i in range(num_clusters):
        topk_values = sort_closest_center(centers[i], clusters, word_embeddings, i)
        # if rerank:
        #     indices.append(find_top_k_words(100, topk_vals, vocab))
        # else:
        indices.append(find_top_k_words(topk, topk_values, corpus))

    return clusters, centers, indices

def get_clusters_kmedoids(word_embeddings, num_clusters, topk, corpus):
    # Fit the kmeans model to the embeddings
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=20).fit(word_embeddings)
    # Return the labels of words and center of each cluster
    clusters = kmedoids.labels_
    centers = np.array(kmedoids.cluster_centers_)

    indices = []

    for i in range(num_clusters):
        topk_values = sort_closest_center(centers[i], clusters, word_embeddings, i)
        # if rerank:
        #     indices.append(find_top_k_words(100, topk_vals, vocab))
        # else:
        indices.append(find_top_k_words(topk, topk_values, corpus))

    return clusters, centers, indices

def get_clusters_spkmeans(word_embeddings, num_clusters, topk, corpus):
    # Fit the spherical kmeans mdoel to the embeddings
    spkmeans = SphericalKMeans(n_clusters=num_clusters, max_iter=10, random_state=20, verbose=0).fit(word_embeddings)
    # Return the labels of words and center of each cluster
    clusters = spkmeans.labels_
    centers = np.array(spkmeans.cluster_centers_)

    indices = []

    for i in range(num_clusters):
        topk_values = sort_closest_center(centers[i], clusters, word_embeddings, i)
        # if rerank:
        #     indices.append(find_top_k_words(100, topk_vals, vocab))
        # else:
        indices.append(find_top_k_words(topk, topk_values, corpus))

    return clusters, centers, indices

# Choosing the dataset to work with

dataset_name = input("Choose the dataset to work with, reuter8 or 20newsgroup or 20newsgrouppre: ")

num_document = int(input("Number of documents desired: "))

def dataset_loader(dataset_name, num_document):

    if dataset_name == 'reuter8':
        docs = pd.read_csv("Reuters_8_classes_test.csv")
        text = docs.text.to_list()[:num_document]
    elif dataset_name == '20newsgroup':
        categories=[
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
    ]
        docs = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
        text = docs.data[:num_document]
    elif dataset_name == '20newsgrouppre':
        docs = pd.read_table("20newsgroup_preprocessed_OCTIS.tsv", header=None)
        text = docs.iloc[:, 0].to_list()[:num_document]
    
    return text

text = dataset_loader(dataset_name, num_document)


# Selecting the model configuration

def checkpoint_loader(step, size):

   step = "step" + str(step)
   _model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-" + str(size) + "m-deduped",
    revision=step,
    cache_dir="./pythia-" + str(size) + "m-deduped/" + step,
  )
   _tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-" + str(size) + "m-deduped",
    revision=step,
    cache_dir="./pythia-" + str(size) + "m-deduped/" + step,
  )
   return _model, _tokenizer

step = input("stepsize from 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and every 1000 step up to 143000: ")
size = input("Available model  size 70, 160, 410: ")

model, tokenizer = checkpoint_loader(step, size)

# Defining padding tokens
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Add padding and truncation pre-embedding
maxlength = int(input("Max_length of each document: "))
batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=maxlength, return_tensors='pt')
batch_outputs = model(**batch_inputs, output_hidden_states=True)

# Creating the corpus
corpus = []
for doc_words in batch_inputs['input_ids']:
    for words in doc_words:
        corpus.append(words)

embeddings = batch_outputs.hidden_states[-1]
embeddings = embeddings.detach().numpy()

embeddings_words = embeddings.reshape(-1, embeddings.shape[-1])

# Removed the padding tokens in the corpus and the corresponding embeddings
corpus_words = tokenizer.batch_decode(corpus)
valid_word_index = [i for i in range(len(corpus_words)) if corpus_words[i] != '[PAD]']
embeddings_words_valid = embeddings_words[valid_word_index]

valid_corpus_words = [corpus_words[i] for i in range(len(corpus_words)) if corpus_words[i] != '[PAD]']
valid_corpus = [corpus[i] for i in range(len(corpus)) if corpus_words[i] != '[PAD]']

# Calculating the term frequency
tf = [valid_corpus_words.count(i)/len(valid_corpus_words) for i in valid_corpus_words]

# Choosing the dimensionality reduction method and reduce the dimensionality of the embeddings in order to reduce the input dimensions
reduced_dim = int(input("Size of each embedding after reduction: "))
reduction_method = input("'pca' for PCA, 'umap' for UMAP: ")
reduction_method_status = False

while reduction_method_status == False:
    if reduction_method == 'pca':
        pca_reducer = PCA(n_components=reduced_dim, random_state=20)
        embeddings_reduced = pca_reducer.fit_transform(embeddings_words_valid)
        reduction_method_status = True
    elif reduction_method == 'umap':
        umap_reducer = umap.UMAP(n_components=reduced_dim, random_state=20)
        embeddings_reduced = umap_reducer.fit_transform(embeddings_words_valid)
        reduction_method_status = True
    else:
        print("please give one of the above mentioned methods")

# Getting the cluster size and selecting the clustering method
num_clusters = int(input("Number of clusters desired: "))
cluster_method = input("'kmeans' for K-means, 'spkmeans' for spherical kmeans, 'kmedoids' for K-Medoids: ")
cluster_method_status = False
num_topk = int(input("Number of top words per cluster: "))

while cluster_method_status == False:
    if cluster_method == 'kmeans':
        cluster_ids, cluster_centers, top_words = get_clusters_kmeans(embeddings_reduced, num_clusters=num_clusters, topk = num_topk, corpus=valid_corpus, weights=tf)
        cluster_method_status = True
    elif cluster_method == 'spkmeans':
        cluster_ids, cluster_centers, top_words = get_clusters_spkmeans(csr_matrix(embeddings_reduced), num_clusters=num_clusters, topk = num_topk, corpus=valid_corpus)
        cluster_method_status = True
    elif cluster_method == 'kmedoids':
        cluster_ids, cluster_centers, top_words = get_clusters_kmedoids(embeddings_reduced, num_clusters=num_clusters, topk = num_topk, corpus=valid_corpus)
        cluster_method_status = True
    else:
        print("Please give one of the above mentioned methods")

k = []
top_words_cleaned = []
for topic in top_words:
    for word in topic:
        k.append(word.replace(" ", ""))
    top_words_cleaned.append(k)
    k = []

print(top_words_cleaned)


for i in top_words_cleaned:
    print(i)

df_results = pd.DataFrame(list(zip(cluster_ids, valid_corpus_words)), columns=['ids', 'word'])
print()
print("The value counts of each cluster: ")
print(df_results["ids"].value_counts())
print()

# For evaluation and comparison with other topic modeling standards
from octis.evaluation_metrics.coherence_metrics import Coherence
# from octis.evaluation_metrics.diversity_metrics import TopicDiversity

data_eval = [doc.split() for doc in text]

topics_eval = []
tp = []
for t in top_words_cleaned:
    for w in t:
        if len(w)>1 and any(w in doc for doc in data_eval):
            tp.append(w)
    if len(tp) == num_topk:
        topics_eval.append(tp)
    tp = []

print(f"Number of clusters that have full top k words: {len(topics_eval)}")


import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary

id2word = corpora.Dictionary(data_eval)
corpus_eval = [id2word.doc2bow(text) for text in data_eval]

coherence_model = CoherenceModel(topics=topics_eval, texts=data_eval, corpus=corpus_eval, dictionary=id2word, coherence='c_v')
coherence_score = round(coherence_model.get_coherence(), 4)

print(f"The coherence score of this topic model is: {coherence_score}")
