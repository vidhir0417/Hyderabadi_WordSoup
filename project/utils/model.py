from nltk.tokenize import PunktSentenceTokenizer 
from utils.utils import vader_wrapper, textblob_wrapper
import pandas as pd


def sentiment(df: pd.DataFrame, 
              col, 
              sentance_mean: bool=False, 
              type_wrapper: str="vader", 
              col_out: str="polarity"):
    '''
    Calculates sentiment analysis for every observation in a column of a DataFrame.

    This function calculates sentiment polairty using Vader or TextBlob over values of a 
    provided column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containg text to analyze.
    - col (Any): Name of a column with text to analyze.
    - sentance_mean (bool): If True function will return average sentiment per sentance, 
                            otherwise returns sentiment of a full observation.
    - type_wrapper (str): Either 'vader' or 'textblob' for type of sentiment analyzer.
    - col_out (str): Name of new column in dataframe with polarity values.

    Returns:
    float: pd.Series with polarity values.
    ''' 

    # tokenize values by sentance if needed.
    df_copy = df.copy()
    if sentance_mean:
        sent_tokenizer = PunktSentenceTokenizer()
        df_copy[col] = df_copy[col].map(lambda review: sent_tokenizer.tokenize(review))

    # calculate polarities depending on analyzer of choice
    if type_wrapper.lower().strip() == "vader":
        vals = df_copy[col].map(lambda review: vader_wrapper(review))
        vals.fillna(0, inplace=True)
        df[col_out] = vals
    elif type_wrapper.lower().strip() == "textblob":
        vals = df_copy[col].map(lambda review: textblob_wrapper(review))
        vals.fillna(0, inplace=True)
        df[col_out] = vals
    else:
        raise ValueError("Unknown wrapper type")
    

    
    return vals

