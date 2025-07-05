import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import json
from youtube_transcript_api import YouTubeTranscriptApi
import string
import contractions
from sklearn.metrics import confusion_matrix, classification_report
from googleapiclient.discovery import build
import time
import os
import sys



def cleaning_sentence(text):
    
    text = text.lower()
    PUNCT_TO_REMOVE = string.punctuation
    ans = contractions.fix(text).translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    ans = ans.replace('music','')
    ans = ans.replace("  ",' ')
    ans = " ".join(ans.split())
    return ans

def process_text(text):
    """
    Removes stop words and lemmatizes the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The processed text with stop words removed and lemmatized.
    """
    
    doc = nlp(text)
    filtered_and_lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text)>2]
    index = 1
    while index<len(filtered_and_lemmatized_tokens):
        if filtered_and_lemmatized_tokens[index] == filtered_and_lemmatized_tokens[index-1]:
            del filtered_and_lemmatized_tokens[index-1]
        else:
            index = index+1

    return " ".join(filtered_and_lemmatized_tokens).replace('\n','')


