from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import word_freq_calculator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from sklearn.cluster import KMeans



def unsupervised_score_calculator(model_list):
    for tuple in model_list:
        #Inertia
        print("Inertia of {}: {}".format(tuple[0],tuple[1].inertia_))
        #Silhouette Score
        print("Silhouette score of {}: {}".format(tuple[0],silhouette_score(tuple[2],tuple[1].labels_)))
        #calinski-harabasz
        print("Calinski-Harabasz score of {}: {}".format(tuple[0],calinski_harabasz_score(tuple[2],tuple[1].labels_)))
        print("\n")


def inertia_plotter(tf_matrix, max_k = 10, verbose=False):
    x_k_nr = []
    y_inertia = []
    for k in tqdm(range(2,max_k+1)):
        x_k_nr.append(k)
        kmeans = KMeans(n_clusters=k,random_state=0).fit(tf_matrix)
        y_inertia.append(kmeans.inertia_)
        if verbose==True:
            print("For k = {}, inertia = {}".format(k,round(kmeans.inertia_,3)))
    fig = px.line(x=x_k_nr, y=y_inertia, markers=True)
    fig.show()


def elbow_finder(tf_matrix, max_k=10, verbose=True):
    
    y_inertia = []
    for k in tqdm(range(1,max_k+1)):
        kmeans = KMeans(n_clusters=k,random_state=0).fit(tf_matrix)
        if verbose==True:
            print("For k = {}, inertia = {}".format(k,round(kmeans.inertia_,3)))
        y_inertia.append(kmeans.inertia_)

    x = np.array([1, max_k])
    y = np.array([y_inertia[0], y_inertia[-1]])
    coefficients = np.polyfit(x, y, 1)
    line = np.poly1d(coefficients)

    a = coefficients[0]
    c = coefficients[1]

    elbow_point = max(range(1, max_k+1), key=lambda i: abs(y_inertia[i-1] - line(i)) / np.sqrt(a**2 + 1))
    print(f'Optimal value of k according to the elbow method: {elbow_point}')
    
    return elbow_point


def cluster_namer_bow(dataset, label_column_name, column_to_name= 'foods', nr_words=5):
    labels = list(set(dataset[label_column_name]))

    # Generate corpus: one document per label
    corpus = []
    for label in labels:
        label_doc = " ".join(dataset[column_to_name].loc[dataset[label_column_name] == label].dropna())
        if label_doc.strip():  # Include only non-empty documents
            corpus.append(label_doc)

    # Handle empty corpus scenario
    if not corpus:
        raise ValueError("No valid documents found in the corpus after preprocessing.")

    # Bag-of-Words vectorization
    bow_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", stop_words=None)
    bow_td_matrix = bow_vectorizer.fit_transform(corpus)
    bow_word_list = bow_vectorizer.get_feature_names_out()

    label_name_list = []
    for idx, _ in enumerate(corpus):
        label_vocabulary = word_freq_calculator(
            bow_td_matrix[idx].toarray(), bow_word_list, df_output=True
        )
        label_vocabulary = label_vocabulary.head(nr_words)
        label_name = "_".join(label_vocabulary["words"])
        label_name_list.append(label_name)

    label_name_dict = dict(zip(labels, label_name_list))
    dataset[label_column_name] = dataset[label_column_name].map(lambda label: label_name_dict.get(label, label))

    return dataset


def plotter_3d_cluster(dataset_org, vector_column_name, cluster_label_name, column_to_name_on_namer, write_html=False, html_name="test.html"):
    
    dataset = dataset_org.copy()
    dataset = cluster_namer_bow(dataset, cluster_label_name,column_to_name_on_namer, nr_words=3)

    svd_n3 = TruncatedSVD(n_components=3)
    td_matrix = np.array([[component for component in doc] for doc in dataset[vector_column_name]])
    svd_result = svd_n3.fit_transform(td_matrix)

    for component in range(3):
        col_name = "svd_d3_x{}".format(component)
        dataset[col_name] = svd_result[:,component].tolist()

    fig = px.scatter_3d(dataset,
                        x='svd_d3_x0',
                        y='svd_d3_x1',
                        z='svd_d3_x2',
                        color=cluster_label_name,
                        title=vector_column_name+"__"+cluster_label_name,
                        opacity=0.7,
                        hover_name = cluster_label_name,
                        color_discrete_sequence=px.colors.qualitative.Alphabet)

    if write_html==True:
        fig.write_html(html_name)
    fig.show()
    

def cluster_namer_doc2vec(dataset, label_column_name, nr_words=5):
    labels = list(set(dataset[label_column_name]))

    # Generate corpus: one document per label
    corpus = []
    for label in labels:
        label_doc = " ".join(dataset["clean_review_lemmatized"].loc[dataset[label_column_name] == label].dropna())
        if label_doc.strip():  # Include only non-empty documents
            corpus.append(label_doc)

    # Handle empty corpus scenario
    if not corpus:
        raise ValueError("No valid documents found in the corpus after preprocessing.")

    # Create TaggedDocument instances for training Doc2Vec model
    tagged_corpus = [TaggedDocument(words=doc.split(), tags=[str(idx)]) for idx, doc in enumerate(corpus)]

    # Train the Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=10)
    doc2vec_model.build_vocab(tagged_corpus)
    doc2vec_model.train(tagged_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # Create label names based on the most similar words to the document vectors
    label_name_list = []
    for idx, doc in enumerate(corpus):
        # Get the document vector
        doc_vector = doc2vec_model.infer_vector(doc.split())
        
        # Find the most similar documents based on cosine similarity
        similar_docs = doc2vec_model.dv.most_similar([doc_vector], topn=nr_words)
        
        # Get the words associated with these similar documents (tags)
        label_name = "_".join([str(similar_docs[i][0]) for i in range(nr_words)])
        label_name_list.append(label_name)

    # Create a dictionary mapping labels to generated names
    label_name_dict = dict(zip(labels, label_name_list))
    
    # Map the label names back to the dataset
    dataset[label_column_name] = dataset[label_column_name].map(lambda label: label_name_dict.get(label, label))

    return dataset


def plotter_3d_cluster_doc2vec(dataset_org, cluster_label_name, nr_words=5, write_html=False, html_name="test.html"):
    
    # Prepare the dataset
    dataset = dataset_org.copy()

    # Generate corpus for Doc2Vec
    labels = list(set(dataset[cluster_label_name]))
    corpus = []
    for label in labels:
        label_doc = " ".join(dataset["clean_review_lemmatized"].loc[dataset[cluster_label_name] == label].dropna())
        if label_doc.strip():  # Include only non-empty documents
            corpus.append(label_doc)

    # Handle empty corpus scenario
    if not corpus:
        raise ValueError("No valid documents found in the corpus after preprocessing.")
    
    # Create TaggedDocument instances for training the Doc2Vec model
    tagged_corpus = [TaggedDocument(words=doc.split(), tags=[str(idx)]) for idx, doc in enumerate(corpus)]

    # Train the Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=10)
    doc2vec_model.build_vocab(tagged_corpus)
    doc2vec_model.train(tagged_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # Generate document vectors using Doc2Vec
    doc_vectors = [doc2vec_model.infer_vector(doc.split()) for doc in corpus]

    # Perform dimensionality reduction to 3D using SVD
    svd_n3 = TruncatedSVD(n_components=3)
    svd_result = svd_n3.fit_transform(np.array(doc_vectors))

    # Add SVD components to dataset
    for component in range(3):
        col_name = "svd_d3_x{}".format(component)
        dataset[col_name] = svd_result[:, component].tolist()

    # Cluster the documents (example: KMeans clustering)
    kmeans = KMeans(n_clusters=len(labels), random_state=42)
    dataset[cluster_label_name] = kmeans.fit_predict(svd_result)

    # Create a 3D scatter plot using Plotly
    fig = px.scatter_3d(dataset,
                        x='svd_d3_x0',
                        y='svd_d3_x1',
                        z='svd_d3_x2',
                        color=cluster_label_name,
                        title="Doc2Vec 3D Clustering: "+cluster_label_name,
                        opacity=0.7,
                        hover_name="clean_review_lemmatized",
                        color_discrete_sequence=px.colors.qualitative.Alphabet)

    # Save plot as HTML file if specified
    if write_html:
        fig.write_html(html_name)

    # Show the plot
    fig.show()



#Acessing the last k variables created
def get_last_n_created_vars(n=10):
    # Get all global variables
    global_vars = globals()

    # Filter the global variables that start with 'cluster_' (assuming they are named that way)
    cluster_vars = [var for var in global_vars if var.startswith('cluster_')]

    # Return the last n created cluster variables
    return cluster_vars[-n:]

def filter_clusters(reviews_dataset, cluster_column_name='kmeans_bow_clusters'):
    # Create a dictionary to store the cluster datasets
    clusters_dict = {}
    
    # Get unique cluster labels from the specified column
    cluster_labels = reviews_dataset[cluster_column_name].unique()
    
    # Loop through each cluster label and create an entry in the dictionary for each cluster
    for label in cluster_labels:
        cluster_name = f"cluster_{label}"  # Create a key name dynamically
        clusters_dict[cluster_name] = reviews_dataset[reviews_dataset[cluster_column_name] == label]
    
    return clusters_dict