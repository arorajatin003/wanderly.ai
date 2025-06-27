import asyncio
import os
import string
import spacy
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    NotTranslatable,
    VideoUnavailable
)
import contractions
from googletrans import Translator

# List of YouTube video IDs to process
list_video_id = [
    # 'p0MvovsCxCk',
    # '7cPLbiblb84',
    # 'rm_j6O8y148',
    # 'Lv0PkSkKeSo',
    # '5zA6OFpkPe0',
    # 'Oz18u64bM8I',
    # 'p0MvovsCxCk',
    # '7cPLbiblb84',
    # 'Wt1LgJyF3s',
    'fP2zols1dag',
    'D1blpY-3ROE',
    'snwAYESRUEw',
    '48TcOu9kPqg',
    'dcRh1zTPnDQ',
    'lmhzpFfuWKc',
    '45qqtIpQP5M',
]

translator = Translator()
nlp = spacy.load("en_core_web_sm")

def cleaning_sentence(text):
    text = text.lower()
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('music', '')
    return " ".join(text.split())

def sync_translate(text, src='auto', dest='en'):
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return ' '.join([translat.text for translat in translated])
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

async def fetch_transcript(video_id):
    """
    Fetch transcript for a given video ID and save cleaned, translated text to file.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try native English first
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US'])
            transcript_text = " ".join([entry.text for entry in transcript.fetch()])
        except NoTranscriptFound:
            # Try translating from another language
            for tr in transcript_list:
                try:
                    ts = tr.fetch()
                    raw_text = " ".join([entry.text for entry in ts])
                    print(f'Translation started for {video_id}')
                    transcript_text = sync_translate([entry.text for entry in ts], src=tr.language_code, dest='en')
                    break
                except Exception as e:
                    print(f"Failed to fetch/translate transcript from {tr.language_code}: {e}")
            else:
                raise Exception("No transcript could be fetched or translated.")

        # Process with spaCy
        doc = nlp(transcript_text.lower().replace("  ", " "))
        cleaned_sentences = [cleaning_sentence(sent.text) for sent in doc.sents if sent.text.strip()]

        # Save to file
        root = os.getcwd()
        file_path = os.path.join(root, 'data', 'transcripts')
        os.makedirs(file_path, exist_ok=True)

        with open(os.path.join(file_path, f'transcript_{video_id}.txt'), "w", encoding='utf-8') as f:
            for sent in cleaned_sentences:
                f.write(sent + '\n')

        print(f"Saved transcript for {video_id}")

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as err:
        print(f"Skipped {video_id}: {err}")
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
    finally:
        print(f"Finished processing {video_id}")

async def fetch_all_transcripts():
    for video_id in list_video_id:
        print(f'starting {video_id}  processing' )
        await fetch_transcript(video_id)

async def main():
    print("Starting transcript fetcher...")
    await fetch_all_transcripts()
    print("All transcripts processed.")

if __name__ == "__main__":
    asyncio.run(main())
