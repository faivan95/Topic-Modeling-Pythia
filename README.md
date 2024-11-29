Topic modeling across pretraining done alongside Dongchen Dai and Andreea Elena Bodea.

Topic modeling is the NLP task of extracting topics from text documents and typically used probabilistic models, such as Latent-Dirichlet Allocation (LDA) etc.
Here topic modeling is done using the more recent and statistically better approach of using clustering methods on Large Language Model embeddings (most popularly BERTopic).

# _GPT-NeoX_ has publicly available weights after each training cycle. The embeddings at different pretraining times were utilized to see if they would lead to better topic modeling results.

Kmeans, kmedoids and Spherical-kmeans were used as the clustering methods. 

Reuter8, 20newsgroup or the preprocessed version of 20newsgroup were used as the datasets for topic modeling to be applied to.

#Due to memory constraints, it is advised to run this code on a server as even 1000 rows of data with maxlength of 100 characters will eat up the entire allocated memory in Google collab once embedded.

Both Principal Component Analysis (PCA) and UMAP are available for embedding dimension reduction but PCA is highly advised as it lead to better results.

#The generated topics were evaluated based on Topic diversity and coherence. Results obtained from utilizing embeddings at different checkpoints varied greatly.
#The findings can be discussed upon request.
