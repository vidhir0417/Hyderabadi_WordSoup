from utils.utils import hashtag_changer, slang_corr  # or utils.utils ?
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer
import nltk
import re
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger_eng')


# create list with all possible values for column
def label_into_lt(data, col):

    '''
    Extract unique values from a column of a DataFrame.

    Parameters:
    - data (DataFrame): The input DataFrame.
    - col (str): The name of the column to extract unique values from.

    Returns:
    - list: A list of unique values from that column.
    '''

    label_list = []
    for idx in range(len(data)):
        for el in data[col].iloc[idx]:
            if el not in label_list:
                label_list.append(el)

    return label_list


# encode values in column into One-Hot-Encoding
def label_encoder_column(local_label_lt, label_lt):

    '''
    Encodes a list of labels into a binary format based on the provided label list.

    Parameters:
    - local_label_lt (list): A list of labels to encode.
    - label_lt (list): A list of the possible/unique labels.

    Returns:
    - list: A binary list indicating the presence (1) or absence(0) of each label in local_label_lt.
    '''

    label2id_multilabel = {label: id for id, label in enumerate(label_lt)}
    # id2label_multilabel = {id:label for label, id in label2id_multilabel.items()}

    encoded_label_lt = [0 for i in range(len(label_lt))]
    for label in local_label_lt:
        label_id = label2id_multilabel[label]
        encoded_label_lt[label_id] = 1
    return encoded_label_lt


def text_cleaner(text: str,
                 lower: bool = False,
                 no_numbers: bool = False,
                 user_id: bool = False,
                 hashtag: bool = False,
                 links: bool = False,
                 punctuation: int = 0,
                 no_double_spaces: bool = False,
                 no_emojis_u8: bool = False,
                 no_emojis_non_u8: bool = False,
                 non_lat_letter: bool = False,
                 slang_abbr: bool = False) -> str:

    '''
    Cleans a piece of text by applying a set of transformations.

    This function takes a text as input and returns a cleaned version of that same text. This is done by applying a set of
    transformations selected by the user, such as: lowercasing, punctuation removal..
    The 'cleaned' text will be useful for our text analysis.

    Parameters:
    - text (str): The input text to be cleaned.
    - lower (bool): If True, convert text to lowercase.
    - no_numbers (bool): If True, remove numbers.
    - user_id (bool): If True, replace user IDs with "user".
    - hashtag (bool): If True, remove hashtags.
    - links (bool): If True, replace links with the word "link".
    - punctuation (int): Level of punctuation removal (0: no punctuation removal, 1:some punctuation removal, or 2: punctuation removal).
    - no_double_spaces (bool): If True, remove double spaces.
    - no_emojis_u8 (bool): If True, remove specific emojis in UTF-8.
    - no_emojis_non_u8 (bool): If True, remove emojis not in UTF-8.
    - non_lat_letter (bool): If True, replace non-Latin characters with their Latin equivalents.
    - slang_abbr(bool): If True, replace slang and abbreviated words to their full meaning.

    Returns:
    str: Cleaned text
    '''

    # DEFINING IMPORTANT PATTERNS 
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F900-\U0001F9FF"  # symbols & pictographs 2.0
                               u"\U0001F100-\U0001F1FF"  # Alphanumeric Supplement
                               "]+", flags=re.UNICODE)

    punctuation_pattern = ("[" +
                           "\u0021-\u0026"
                           "\u0028-\u002C"
                           "\u002E-\u002F"
                           "\u003A-\u003F"
                           "\u005B-\u005F"
                           "\u2010-\u2028"
                           "\ufeff\u0023\u0040"
                           "\-`{|}~™₹▪️÷-"
                           "]+")

    # Captures apostrophes that are not part of contractions (apostrophes followed or preceded by a whitespace)
    apostrophe_pattern = r"(?<=\s)[\'\u2018\u2019\u02BC\u0060]|[\'\u2018\u2019\u02BC\u0060](?=\s)"
    # in this one ? (\u003F), ! (\u0021), < (\u003C), > (\u003E), and $ (\u0024) have been excluded.
    punctuation_pattern_for_sentiment_analysis = ("[\u0022-\u0024"
                                                  "\u0026"
                                                  "\u0028-\u002C"
                                                  "\u002E-\u002F"
                                                  "\u003A-\u003C"
                                                  "\u003E\u003F"
                                                  "\u005B-\u005F"
                                                  "\u2010-\u2028"
                                                  "\ufeff\u0023\u0040"
                                                  "\-`{|}~™₹▪️÷-]+")

    link_pattern = (
                    "[A-Za-z0-9]+\@[a-z]+\.com|" +  # email
                    "https{0,1}\://www\.[A-Za-z0-9\.]+\.com\/[A-Za-z0-9]+|" +  # https://www.novaims.pt.com/memes
                    "https{0,1}\://[A-Za-z0-9\.]+\.com/[A-Za-z0-9]+|" +  # https://novaims.pt.com/memes
                    "www\.[A-Za-z0-9\.]+\.com/[A-Za-z0-9]+|" +  # www.novaims.com/memes
                    "[A-Za-z0-9\.]+[A-Za-z0-9]+\.com"  # novaims.pt.com
                    )

    text = str(text).replace("\n", " ").replace("\r", " ")

    if non_lat_letter:
        text = text.replace("à", "a")
        text = text.replace("è", "e")
        text = text.replace("é", "e")
        text = text.replace("ñ", "n")
        text = text.replace("ú", "u")
        text = text.replace("ä", "a")
        text = text.replace("À", "A")
        text = text.replace("ģ", "g")
        text = text.replace("ś", "s")
        text = text.replace("ç", "c")

    if slang_abbr:
        text = re.sub(re.compile("\b[uU]\b"), " you ", text)
        text = re.sub(re.compile("[tT][hH][nN][xX]"), " thanks ", text)
        text = re.sub(re.compile("[gG][uU][dD]"), " good ", text)
        text = re.sub(re.compile("[kK][uU][dD][oO][sS]"), " praise ", text)
        text = re.sub(re.compile("\b[nN]\b"), " and ", text)
        text = re.sub(re.compile("\b[rR]\b"), " are ", text)
        text = re.sub(re.compile("\b[rR][sS]\b"), " rupees ", text)
        text = re.sub(re.compile("[bB][dD][aA][yY]"), " birthday ", text)
        text = re.sub(re.compile("[tT][bB][hH]"), " to be honest ", text)
        text = re.sub(re.compile("\b[lL][iI][lL]\b"), " little ", text)
        text = re.sub(re.compile("[vV][rR][yY]"), " very ", text)
        text = re.sub(re.compile("[qQ][tT][yY]"), " quantity ", text)
        text = re.sub(re.compile("\b[eE][sS][pP]\b"), " especially ", text)
        text = re.sub(re.compile("\b[pP][lL][szSZ]\b"), " please ", text)
        text = re.sub(re.compile("[nN][yY][cC]"), " nice ", text)
        text = re.sub(re.compile("\b[cC][oO][zZ]\b"), " because ", text)
        text = re.sub(re.compile("\b[dD][eE][cC][oO][rR]\b"), " decoration ", text)
        text = re.sub(re.compile("\b[nN][tT]\b"), " not ", text)
        text = re.sub(re.compile("\b[dD][iI][nN][tT]\b"), " didn't ", text)

    if no_numbers:
        text = re.sub(re.compile("[0-9]"), " ", text)

    if no_emojis_u8:
        # getting rid of basic emojis that appear in utf-16
        text = re.sub(re.compile("[\u2600-\u27FF\u200d\u2B00-\u2B7F\uFE0E\u20E3]"), " ", text)

    if no_emojis_non_u8:
        # getting rid of emojis that don't appear in utf-16
        text = re.sub(emoji_pattern, " ", text)

    if user_id:
        # pattern that starts with newline/space/start of string being followed by @
        # and then combination of word chars
        # user is lower so it doesn't affect sentiment
        text = re.sub(re.compile("(^|\s|\n)@[A-Za-z0-9_]+"), " user", text)

    if hashtag:
        # check utils.hashtag_changer
        text = hashtag_changer(text)

    if links:
        text = re.sub(re.compile(link_pattern), "link", text)

    if punctuation == 0:
        pass
    elif punctuation == 1:
        text = re.sub(punctuation_pattern_for_sentiment_analysis, " ", text)
        text = re.sub(apostrophe_pattern, " ", text)
    elif punctuation == 2:
        # remove all punctuation
        text = re.sub(punctuation_pattern, " ", text)
        text = re.sub(apostrophe_pattern, " ", text)

    if no_double_spaces:
        text = re.sub(re.compile("[\s]{2,}"), " ", text)

    if lower:
        text = text.lower()

    return text


# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag: str) -> str:

    '''
    Maps NLTK POS tags to WordNet POS tags (so we can use it on lemmatize_all function)

    Parameters:
    - tag (str): The part-of-speech tag from NLTK.

    Returns:
    - str: The corresponding WordNet POS tag, or 'None' if no mapping is found.
    '''

    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Lemmatization function
def lemmatize_all(tokenized_text: list, pos_tags: list = None) -> list:

    '''
    Performs lemmatization to the provided tokenized text.

    Parameters:
    - tokenized_text (list): A list of tokenized words.
    - pos_tags (list): A list of part-of-speech tags corresponding to the tokens (By default, the nltk.pos_tag will be used).

    Returns:
    - list: A list of lemmatized tokens.
    '''

    wordnet_lem = nltk.stem.WordNetLemmatizer()
    if pos_tags is None:
        pos_tagged = pos_tag(tokenized_text) 
    else:
        pos_tagged = [(tok, pos) for tok, pos in zip(tokenized_text, pos_tags)]

    lemmatized_tokens = []
    for token, tag in pos_tagged:
        if token.lower() == "us":  
            # Prevent lemmatization of 'us' to 'u'
            lemmatized_token = token
        else:
            wordnet_tag = get_wordnet_pos(tag)
            if wordnet_tag:  
                # If there is a relevant POS tag, lemmatize with it
                lemmatized_token = wordnet_lem.lemmatize(token, wordnet_tag)
            else:
                # Otherwise, lemmatize as a noun (default)
                lemmatized_token = wordnet_lem.lemmatize(token)
        lemmatized_tokens.append(lemmatized_token)

    return lemmatized_tokens


def pipeline(df_,
             new_col_name: str = "clean_text",
             tokenize: bool = False,
             pos_tag_: bool = False,
             no_stopwords: bool = False,
             stops_to_keep: set = {"not", "no", "never"}, 
             words_to_drop: set = {},
             lemmatized: bool = False,
             contractions: bool = False,
             **kwargs):

    '''
    Main pipeline for preprocessing.

    It receives the reviews dataframe and returns a dataframe with new columns, one of them being the cleaned text and the rest being
    the cleaned text with an additional preprocessing text for each one.

    Parameters:
    - df (DataFrame): The input DataFrame containing the 'Review' column.
    - new_col_name (str): The name of the column for cleaned text.
    - tokenize (bool): If True, tokenize the text.
    - pos_tag_ (bool): If True, perform Part-of-Speech tagging.
    - no_stopwords (bool): If True, remove stopwords.
    - keep_negation (bool): If True, {"not", "no", "never"} will not be removed with other stopwords.
    - lemmatized (bool): If True, perform lemmatization.
    - kwargs: Additional arguments passed to the text_cleaner function.

    Returns:
    - DataFrame: The updated DataFrame with new columns for cleaned text and additional preprocessing steps.
    '''

    df = df_.copy()

    # Clean text
    df[new_col_name] = df.Review.apply(lambda x: text_cleaner(x, **kwargs))

    # Tokenising text
    df[new_col_name + "_token"] = df[new_col_name].apply(lambda x: TweetTokenizer().tokenize(x))

    # Stopwords removal
    if no_stopwords:
        stop_words = set(nltk.corpus.stopwords.words("english"))
        if stops_to_keep:
            # Create a custom stopword list that excludes negation words
            stop_words = stop_words - stops_to_keep

        if words_to_drop:
            stop_words = stop_words + stops_to_keep

        df[new_col_name + "_no_stopwords"] = \
            df[new_col_name + "_token"].apply(lambda x: [item for item in x if item.lower() not in stop_words])

    # POS tagging
    if pos_tag_:
        if new_col_name + "_no_stopwords" in df.columns:
            df[new_col_name + "_pos"] = \
                df[new_col_name + "_no_stopwords"].apply(lambda tokens: [pos[1] for pos in pos_tag(tokens)])
        else:
            df[new_col_name + "_pos"] = \
                df[new_col_name + "_token"].apply(lambda tokens: [pos[1] for pos in pos_tag(tokens)])

    # Lemmatization
    if lemmatized:
        if new_col_name + "_pos" not in df.columns:
            if new_col_name + "_no_stopwords" in df.columns:
                df[new_col_name + "_lemmatized"] = \
                    df[new_col_name + "_no_stopwords"].apply(lambda tokens: lemmatize_all(tokens))
            else:
                df[new_col_name + "_lemmatized"] = \
                    df[new_col_name + "_token"].apply(lambda tokens: lemmatize_all(tokens))

        else: 
            if new_col_name + "_no_stopwords" in df.columns:
                df[new_col_name + "_lemmatized"] = \
                    df.T.apply(lambda line: lemmatize_all(line[new_col_name + "_no_stopwords"],
                                                          line[new_col_name + "_pos"]))
            else:
                df[new_col_name + "_lemmatized"] = \
                    df.T.apply(lambda line: lemmatize_all(line[new_col_name + "_token"],
                                                          line[new_col_name + "_pos"]))
                
    if contractions:
        if new_col_name + "_pos" in df.columns:
            df[new_col_name + "_no_contractions"] = \
                    df.T.apply(lambda line: slang_corr(pos_tags=[(i, j) for (i, j) in 
                                                                 zip(line[new_col_name + "_token"], 
                                                                     line[new_col_name + "_pos"])]))
        elif new_col_name + "_no_stopwords" in df.columns:
            df[new_col_name + "_no_contractions"] = \
                    df.T.apply(lambda line: slang_corr(text=line[new_col_name + "_no_stopwords"]))
        else:
            df[new_col_name + "_no_contractions"] = \
                    df.T.apply(lambda line: slang_corr(text=line[new_col_name + "_token"]))
                  
    # Detokenizing if tokenize=False
    if not tokenize:
        if new_col_name + "_no_stopwords" in df.columns:
            df[new_col_name + "_no_stopwords"] = \
                df[new_col_name + "_no_stopwords"].apply(lambda x: " ".join(x))
        if new_col_name + "_lemmatized" in df.columns:
            df[new_col_name + "_lemmatized"] = \
                df[new_col_name + "_lemmatized"].apply(lambda x: " ".join(x))

        df.drop(columns={new_col_name + "_token"}, inplace=True)

    return df
