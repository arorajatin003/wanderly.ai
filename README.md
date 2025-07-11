
# Wanderly.ai

## 📌 About This Project
This project is an NLP-powered pipeline for extracting meaningful keywords and topics from YouTube video transcripts—specifically tailored for travel-related content. It identifies named entities like landmarks, foods, and local experiences, and builds a personalized recommendation engine based on user preferences.

The system processes transcripts, trains a custom NER model, clusters extracted entities using NMF (Non-negative Matrix Factorization), and recommends keywords relevant to the user's travel interests.

---

## 🎯 Scope
- Download and clean YouTube video transcripts
- Train a custom Named Entity Recognition (NER) model
- Extract, rank, and group key concepts and travel-specific entities
- Recommend keywords and experiences using cosine similarity
- Provide a modular pipeline with reusable components (NER, TF-IDF, NMF, similarity scoring)

---

## ⚙️ How to Run the Project

Ensure your environment has the necessary Python libraries installed (`spaCy`, `scikit-learn`, `pandas`, `youtube_transcript_api`, `google-api-python-client`, etc.)

### Step 1: Save Transcripts from YouTube
Use the following script to fetch and store video transcripts:
```bash
python api.py
```

This will use `YouTubeTranscriptApi` to download subtitles and save them to the `data/transcripts/` folder.

---

### Step 2: Train the NER Model
To train the custom Named Entity Recognition model:
```bash
python model_training.ipynb
```

This trains a spaCy NER model using labeled data and saves the model in the `models/ner_mod/model-best/` directory.

---

### Step 3: Extract Keywords & Generate Recommendations
Run the main processing and recommendation script:
```bash
python key_word_extraction.ipynb
```

This script:
- Loads the transcripts
- Applies the trained NER model
- Extracts relevant phrases and entities
- Applies topic modeling (NMF) and TF-IDF vectorization
- Recommends the top relevant travel experiences or keywords for the user

---

## 📁 Folder Structure

```
.
├── config                  # NER model training congigs
├── api.py                  # Transcript fetch logic
├── model_training.py       # NER model training
├── key_word_extraction.py  # Keyword extraction & recommendation
├── data/
│   └── transcripts/        # Saved transcripts
├── models/
│   └── ner_mod/            # Trained NER model
```

---

## ✅ Dependencies
- spaCy
- scikit-learn
- youtube-transcript-api
- google-api-python-client
- pandas, numpy, contractions, tqdm, etc.

---

## 👨‍💻 Author
Wanderly NLP Team

For questions or collaboration, please contact: [arorajatin003@example.com]
