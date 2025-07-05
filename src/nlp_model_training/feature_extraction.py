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
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

def get_video_statistics(video_id): 
    # video_id = '7cPLbiblb84'
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        request = youtube.videos().list(
            part=['snippet','statistics'],
            id=video_id
        )
        response = request.execute()
        stats = response['items'][0]['statistics']
        viewCount = stats['viewCount'] if 'viewCount' in stats.keys() else 0
        likeCount = stats['likeCount'] if 'likeCount' in stats.keys() else 0
        commentCount = stats['commentCount'] if 'commentCount' in stats.keys() else 0 
        channelId = response['items'][0]['snippet']['channelId']
        channel_name = response['items'][0]['snippet']['channelTitle']

        print(response['items'][0]['statistics'],channelId,channel_name)
        
        return int(viewCount),int(likeCount),int(commentCount),channel_name,channelId
    except Exception as e:
        print(e)

def get_channel_statistics(channelId):
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        request_channel =  youtube.channels().list(
            part=['statistics'],
            id=channelId
        )
        response_channel = request_channel.execute()
        subscriberCount= response_channel['items'][0]['statistics']['subscriberCount']
        return int(subscriberCount)
    except Exception as e:
        print(e)

def get_landmarks(video_ids):
    root = os.getcwd()
    model_path = os.path.join(root.split('notebooks')[0], 'models', 'ner_mod','model-best')
    nlp_ner = spacy.load(model_path)
    nlp     = spacy.load("en_core_web_lg")
    ner     = dict()
        
    for video_id in video_ids:
        try:
            print(video_id)
            
            viewCount,likeCount,commentCount,channel_name,channelId = get_video_statistics(video_id)
            subscriberCount = get_channel_statistics(channelId)

            root = os.getcwd()
            file_path = os.path.join(root.split('notebooks')[0], 'data', 'transcripts')

            file = open(f'{file_path}/transcript_{video_id}.txt',"r")
            file_c = open(f'{file_path}/transcript_{video_id}.txt',"r")
            file_string = file_c.read()
            for text in file:
                doc_ner = nlp_ner(cleaning_sentence(text)) 

                for ent in doc_ner.ents:
                    # print(ent,'->', ent.label_)
                    
                    doc = nlp(ent.text)

                    rating = file_string.count(ent.text) + ((viewCount+likeCount+commentCount+subscriberCount)/(subscriberCount))
            
                    lemma = ' '.join([token.lemma_ for token in doc])
                    if lemma in ner.keys():
                        # ner[ent.label_].append(ent.text)
                        ner[lemma]['ENT'].add(ent.label_)
                        ner[lemma]['rating'] = (rating+ner[lemma]['rating'])/2
                        ner[lemma]['suggested_by'].add(channel_name)
                        ner[lemma]['video_id'].add(video_id)
                    else:
                        # ner[ent.label_] = [ent.text]
                        ner[lemma] = {
                            'word': ent.text,
                            'ENT': set([ent.label_]),
                            # 'keywords': list,
                            'suggested_by': set([channel_name]),
                            'rating': rating,
                            'video_id':set([video_id])
                        }
                # print('--x-x-x--x-x-x-x-x-x-x---x-x-x--')
        except Exception as e:
            print(e)
    
    df = pd.DataFrame(ner)
    df = df.T
    df = df.reset_index().rename(columns={'index':'lemma_word'})

    df['ENT_len'] = df['ENT'].apply(lambda x: len(list(x))) 
    df['ENT_max'] = df['ENT'].apply(lambda x: (list(x))[0])

    df[df['lemma_word']!=df['word']]
    df.to_csv('D:/randomProjects/wanderly.ai/data/Parquet/all_locations_&_food.csv')
    
