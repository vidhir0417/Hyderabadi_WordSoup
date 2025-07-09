import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import defaultdict
from collections import Counter
from sklearn import metrics
from tqdm import tqdm
tqdm.pandas()
import networkx as nx
import spacy


def fold_score_calculator(y_pred, y_test, verbose=False):
    
    #6. Compute the binary classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    if verbose == True:
        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(acc,prec,recall,f1))
    return (acc, prec, recall, f1)


def hashtag_changer(text):

    '''
    Returns text with modified hashtags in the given text for better readability.

    This functions takes text as input as returns a text with modified hashtags.
    It identifies normal hastags and modifies them by splitting camel case, alphanumeric or mixed-case hastags into separate words.
    Hashtags with entire uppercase or lowercase remain unchanged. The modified hashtags are replaced with the original ones.

    Parameters:
    - text (str): The input text containing the original hashtags.

    Returns:
    - str: modified text with improved hashtage readability.
    '''

    # find all hashtags followed with word characters
    hashtags = re.findall(re.compile("(#\w+)"), text)
    # list for hashtag after partition
    new_hashtags = []
    for hashtag in hashtags:
        # drop "#" for each found hashtag
        hashtag = hashtag.replace("#", "")
        if (len(re.sub(re.compile("[A-Z]+"), "", hashtag)) != 0 and
            len(re.sub(re.compile("[a-z]+"), "", hashtag)) != 0):
            # if hashtag is not all capital or lower case letter find all words of following patterns in it
            # lower case letters (e.g. chicken in #hungry4chicken),
            # capital start follow by lower (e.g. Nice in #VeryNice) and numbers (e.g. 4 in #hungry4chicken)
            words = re.findall(re.compile("([a-z]+|[A-Z][a-z]{0,}|[0-9]+)"), hashtag)
            # a loop to combine all parts of separated hashtag (e.g. "hungry", "4", "chicken" in #hungry4chicken)
            final = " " 
            for word in words:
                final += word + " "
            new_hashtags.append(" " + final)
        else:
            # if hashtag is all capital or lower case we just append it as is
            new_hashtags.append(" " + hashtag)
    # replacing old hashtags with new hashtags
    for ho, hn in zip(hashtags, new_hashtags):
        text = text.replace(ho, hn)
    return text


def word_freq_calculator(td_matrix, word_list, df_output=True):

    '''
    Returns the word frequency from a term-document matrix.

    This function takes a term-document matrix and a list of words as input.
    It returns either a dictionary or a DataFrame of the word count.
    The word count is the total amount of words based on its occurrence on all the documents in the providded term-document matrix.

    Parameters:
    - td_matrix (numpy.ndarray): A matrix of vectors
    - word_list (list): A list with the names of the words.
    - df_output (bool, optional): If `True`, a DataFrame sorted by frequency. IF `False`, a dictionary. Default is `True`.

    Returns:
    pandas.DataFrame or dict: A DataFrame with each word and its respective count if `df_output` is True, otherwise a dictionary with the same content.
    '''

    word_counts = np.sum(td_matrix, axis=0).tolist()
    if df_output == False:
        word_counts_dict = dict(zip(word_list, word_counts))
        return word_counts_dict
    else:
        word_counts_df = pd.DataFrame({"words": word_list, "frequency": word_counts})
        word_counts_df = word_counts_df.sort_values(by=["frequency"], ascending=False)
        return word_counts_df
  
    
def plot_term_frequency(df, nr_terms, df_name, show=True, figsize=(10, 8)):

    '''
    Returns the figure with the plotted term frequencies from a given DataFrame.

    This function takes a DataFrame and the number of terms as input.
    It returns a plot of the top most frequent words in a DataFrame.
    Using a Seaborn barplot, the terms are on the y-axis and its corresponding frequencies on the x-axis.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the terms and their counts.
    - nr_terms (int): The number of top terms to include in the figure.
    - df_name (str): The dataset name to be used for the figure's title.
    - show (bool, optional): If `True`, figure is displayed, else the figure is returned without display. Default is `True`.

    Returns:
    matplotlib.figure.Figure: figure object if `show=False`.
    '''

    # Create the Seaborn bar plot
    plt.figure(figsize=figsize)
    sns_plot = sns.barplot(x='frequency', y='words', data=df.head(nr_terms))  # Plotting top 20 terms for better visualization
    plt.title('Top 20 Term Frequencies of {}'.format(df_name))
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    if show==True:
        plt.show()

    fig = sns_plot.get_figure()
    plt.close()

    return fig


def get_pos_tag(word):

    '''
    Returns the POS tag of a given word.

    This function takes a word as input and returns its POS tag using the `nltk.pos_tag` method.

    Parameters:
    - word (str): The input word.

    Returns:
    str: POS tag of the input word.
    '''

    tag = pos_tag([word])[0][1]
    return tag


def vader_wrapper(user_review):

    '''
    Returns VADER-polarity of passed text.

    This function takes a text or a list of strings as input and returns a VADER polairty value for it.
    If `user_review` is a text the returned polaity will be calculated on it, otherwise if it a list
    returned polarity will be the average of polarities of passed strings.

    Parameters:
    - user_review (str or list): The input text or list of texts.

    Returns:
    float: polarity of passed strings.
    '''

    # initialising VADER analyzer
    vader = SentimentIntensityAnalyzer()
    if type(user_review) == list:
        # finding polarity for each string in a list
        sent_compound_list = []
        for sentence in user_review:
            sent_compound_list.append(vader.polarity_scores(sentence)["compound"])
        # finding average polarity
        polarity = np.array(sent_compound_list).mean()
    else:
        # finding polarity for a single string
        polarity = vader.polarity_scores(user_review)["compound"]
    return polarity


def textblob_wrapper(user_review):

    '''
    Returns TextBlob-polarity of passed text.

    This function takes a text or a list of strings as input and returns a TextBlob polairty value for it.
    If `user_review` is a text the returned polaity will be calculated on it, otherwise if it a list
    returned polarity will be the average of polarities of passed strings.

    Parameters:
    - user_review (str or list): The input text or list of texts.

    Returns:
    float: polarity of passed strings.
    '''

    if type(user_review) == list:
        # finding polarity for each string in a list
        sent_compound_list = []
        for sentence in user_review:
            sent_compound_list.append(TextBlob(sentence).sentiment.polarity)
        # finding average polarity
        polarity = np.array(sent_compound_list).mean()
    else:
        # finding polarity for a single string
        polarity = TextBlob(user_review).sentiment.polarity
    return polarity


def cooccurrence_matrix_window_generator(preproc_sentences, window_size):

    '''
    Returns a co-occurrence matrix for the words that appear in the preprocessed sentences using a sliding window approach.

    This function takes preprocessed sentences and a window size as input.
    It returns a co-occurence matrix for the `preproc_sentences`.
    The entries of the matrix will represent the frequency of the co-occurence of words within the defined window size.

    Parameters:
    - preproc_sentences (list): The input list of tokenized sentences where each sentence is a list of words.
    - window_size (int): The defined input size of the sliding window to determine the co-occuring words.

    Returns:
    pandas.DataFrame: co-occurrence matrix with words as rows and columns and the values as co-occurrence counts.
    '''

    co_occurrences = defaultdict(Counter)

    # Compute co-occurrences
    for sentence in tqdm(preproc_sentences):
        for i, word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    co_occurrences[word][sentence[j]] += 1

    # Ensure that words are unique
    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    # Return the co-occurrence matrix
    return co_matrix_df


def cooccurrence_network_generator(cooccurrence_matrix_df, n_highest_words, output=None):

    '''
    Returns a co-occurence network graph from a co-occurence matrix.

    This functions takes a co-occurence matrix and the number of top words as input and returns a co-occurrence network graph.
    The network graph will only display the top `n_highest_words` based on the total co-occurence words.
    Its nodes will represent words and its edges will represent the co-occurrence relationships with weights corresponding to co-occurence frequency.

    Parameters:
        - cooccurrence_matrix_df (pandas.DataFrame): A co-occurrence matrix.
        - n_highest_words(int): The number of top words to consider.
        - output(str, optional): If "return", returns the figure object, else it displays the graph without returning it. Default is `None`.

    Returns:
    matplotlib.figure.Figure or None: figure object if `output` is "return", otherwise network graph visualizing the co-occurrences.
    '''

    # Filter the top n_highest_words
    filtered_df = cooccurrence_matrix_df.iloc[:n_highest_words, :n_highest_words]

    # Create the graph
    graph = nx.Graph()

    # Add nodes and their sizes
    for word in filtered_df.columns:
        graph.add_node(word, size=filtered_df[word].sum())

    # Add weighted edges
    for word1 in filtered_df.columns:
        for word2 in filtered_df.columns:
            if word1 != word2 and filtered_df.loc[word1, word2] > 0:  # Avoid self-loops and zero-weight edges
                graph.add_edge(word1, word2, weight=filtered_df.loc[word1, word2])

    # Initialize figure (slightly smaller size)
    figure = plt.figure(figsize=(12, 10))

    # Generate positions for nodes with adequate spread
    pos = nx.spring_layout(graph, k=0.5)

    # Scale edge widths (even thinner) and node sizes
    edge_weights = [0.01 * graph[u][v]['weight'] for u, v in graph.edges()]  # Slightly thinner edges
    node_sizes = [max(20, data['size'] / 8) for _, data in graph.nodes(data=True)]  # Same node sizes

    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=edge_weights, alpha=0.6)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')

    # Display the graph
    plt.title(f"Co-occurrence Network ({n_highest_words} words)", fontsize=14)
    plt.show()

    # Optionally return the figure
    if output == "return":
        return figure

  
def slang_corr(text=None, pos_tags = None):
    '''
    Corrects specific slang terms, based on the following word, in given text into original English words.

    This function takes text and Part-of-Speech tags as input.
    It returns text with proper base words in English language.

    Parameters:
    - text (Any): The input text to be corrected.
    - pos_tags (list): If `True`, perform Part-of-Speech tagging. Default is `True`.

    Returns:
    str: Corrected text with no slang depending on the following word.
    '''
    if pos_tags is None:
        if type(text) == str:
            tokens = TweetTokenizer().tokenize(text)
            pos_tags = pos_tag(tokens)
        else:
            pos_tags = pos_tag(text)

    new_tokens = []
    for i, (word, _) in enumerate(pos_tags):
        word_lower = word.lower()
        if word_lower == "ur":
            next_word_tag = pos_tags[i + 1][1] if i < len(pos_tags) - 1 else None
            if next_word_tag == "NN":
                new_tokens.append("your")
            elif next_word_tag in ["VB", "VBG", "DT", "JJ", "RB", "IN"]:
                new_tokens.append("you")
                new_tokens.append("are")
            else:
                new_tokens.append("ur")
        elif word_lower == "v":
            next_word_tag = pos_tags[i + 1][1] if i < len(pos_tags) - 1 else None
            if next_word_tag in ["VB", "VBP", "VBD", "RB"]:
                new_tokens.append("we")
            elif next_word_tag in ["RB", "JJ"]:
                new_tokens.append("very")
            else:
                new_tokens.append("v")
        elif "'s" in word_lower:
            next_word_tag = pos_tags[i + 1][1] if i < len(pos_tags) - 1 else None
            if next_word_tag == "VBG":
                new_tokens.append(word.split("'")[0])
                new_tokens.append("is")
            elif next_word_tag == "VBZ":
                new_tokens.append(word.split("'")[0])
                new_tokens.append("has")
            else:
                new_tokens.append(word)
        elif "'m" in word_lower:
            new_tokens.append(word.split("'")[0])
            new_tokens.append("am")
        elif "'re" in word_lower:
            new_tokens.append(word.split("'")[0])
            new_tokens.append("are")
        elif "'ve" in word_lower:
            new_tokens.append(word.split("'")[0])
            new_tokens.append("have")
        elif "'ll" in word_lower:
            new_tokens.append(word.split("'")[0])
            new_tokens.append("will")
        elif "n't" in word_lower:
            new_tokens.append(word.split("n't")[0])
            new_tokens.append("not")
        else:
            new_tokens.append(word)

            text = " ".join(new_tokens)

    return text

  
# Function Load spaCy model
nlp = spacy.load("en_core_web_sm")
def extract_nouns_and_entities(texts):
    dish_candidates = []
    for text in texts:
        doc = nlp(text.lower())
        for token in doc:
            if token.pos_ == "NOUN":  # Focus on nouns 
                dish_candidates.append(token.text)
    return dish_candidates
