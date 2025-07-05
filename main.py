from typing import Union
import pickle
from fastapi import FastAPI
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse


app = FastAPI()


def get_recommendations():
    user_input = ['I like to trave to find me, enjoy, capture the movement',
    'Like to explore the culture, heritage and local food',
    'relax chill with friends',
    'peacefull place calm and relaxing',
    'chill good local food instagram worthy locations, vloging',
    'Explorer – I love wandering and getting lost',
    'Explore the surroundings on foot',
    'Dream travel worule be lost in the movement, relax, local food, chill vibes, haritage, mountain person but \
    also like beaches and water side resorts']

    user_input = ' '.join(user_input)
    print(user_input)

    user = pd.DataFrame({'name':['Jatin'],'desc':[user_input]})

    user['clean_dec'] = user['desc'].apply(lambda x: process_text(cleaning_sentence(x)))

    print('Done 1')

    X_locations = tvid_loc.transform(user['clean_dec'])
    tvid_loc_df = pd.DataFrame(X_locations.toarray(),columns=tvid_loc.get_feature_names_out())

    print('Done 2')

    X_food = tvid_food.transform(user['clean_dec']) 
    tvid_food_df = pd.DataFrame(X_food.toarray(),columns=tvid_food.get_feature_names_out())

    print('Done 3')

    X_keyWords = tvid_keyWords.transform(user['clean_dec'])
    tvid_keyWords_df = pd.DataFrame(X_keyWords.toarray(),columns=tvid_keyWords.get_feature_names_out())
    # tvid_keyWords_df = pd.DataFrame(X_keyWords.toarray(),columns=tvid_keyWords.get_feature_names_out())
    print('Done 4')
    W_locations = nmf_model_loc.transform(X_locations)
    H_locations = nmf_model_loc.components_
    print('Done 5')
    W_food = nmf_model_food.transform(X_food)
    H_food = nmf_model_food.components_

    W_locations = pd.DataFrame(W_locations,columns=['Forts','Generic','Temple','Temples & Architectur','Scientific Monuments','Lakeside Forts','Views'])

    # H = pd.DataFrame(H,index=['Forts','Generic','Temple','Temples & Architectur','Scientific Monuments','Lakeside Forts','Views'])
    W_food = pd.DataFrame(W_food,columns=['Snaks','Evening','Desert','Tikka'])

    user_transform = pd.concat([user,tvid_keyWords_df,W_locations,W_food],axis=1)
    return user_transform

    

with open('D:/randomProjects/wanderly.ai/models/tvid_loc.pkl', 'rb') as file:
    tvid_loc = pickle.load(file)

with open('D:/randomProjects/wanderly.ai/models/tvid_food.pkl', 'rb') as file:
    tvid_food = pickle.load(file)
with open('D:/randomProjects/wanderly.ai/models/tvid_keyWords.pkl', 'rb') as file:
    tvid_keyWords = pickle.load(file)

with open('D:/randomProjects/wanderly.ai/models/nmf_model_loc.pkl', 'rb') as file:
    nmf_model_loc = pickle.load(file)

with open('D:/randomProjects/wanderly.ai/models/nmf_model_food.pkl', 'rb') as file:
    nmf_model_food = pickle.load(file)

def cleaning_sentence(text):
    text = text.lower()
    PUNCT_TO_REMOVE = string.punctuation
    ans = contractions.fix(text).translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    ans = ans.replace('music','')
    ans = ans.replace("  ",' ')
    ans = " ".join(ans.split())
    return ans

nlp = spacy.load("en_core_web_lg")
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



def get_similar_objects(check,X,Y,top_n=5):
    common = [i for i in X.columns if i in check.columns]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[common])
    user_scaled = scaler.transform(check[common])
    
    # Compute cosine similarity between user vector and all item vectors
    similarity_scores = cosine_similarity(user_scaled, X[common])[0]

    # Add scores to DataFrame
    Y['similarity'] = similarity_scores

    # Sort top recommendations
    top_recommendations = Y.sort_values(by='similarity', ascending=False).head(10)
    return top_recommendations



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/get_recommendations")
def read_item(recommendation_type: Union[str, None] = None):
    df_base_info = pd.read_csv('D:/randomProjects/wanderly.ai/data/Parquet/final_loc_food_df.csv')
    if(recommendation_type is not None):
        df_base_info = df_base_info[df_base_info['ENT_max']==recommendation_type]
    req_col = [i for i in df_base_info.columns if i not in ['lemma_word', 'word', 'suggested_by', 
                                                        'video_id','key_word_extracted','details', 
                                                        'clean_test', 'rating', 'ENT_max','normalize_rating']]
    X = df_base_info[req_col]
    Y = df_base_info[['lemma_word']]

    top_n_recommendations = get_similar_objects(get_recommendations(),X,Y,10)
    top_n_recommendations = df_base_info.merge(top_n_recommendations,on=['lemma_word'],how='inner')[['word','suggested_by','video_id','normalize_rating','ENT_max']]
    top_n_recommendations.rename(columns={'word':'name','normalize_rating':'rating','ENT_max':'type'},inplace=True)
    print(top_n_recommendations.to_dict(orient="records"))
    return JSONResponse(content=top_n_recommendations.to_dict(orient="records"), status_code=200)

@app.get("/x/{type}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}