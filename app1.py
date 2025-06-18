from flask import Flask, request, jsonify
import whisper
import nltk
import os
import traceback
from flask import send_file 
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models only once
whisper_model = whisper.load_model("medium")
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ====== Helper Functions ======


def preprocess_text(text):
    return sent_tokenize(text)

def build_similarity_matrix(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return cosine_similarity(X)

def textrank_summarize(text, top_n=3):
    sentences = preprocess_text(text)
    sim_matrix = build_similarity_matrix(sentences)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return " ".join([sent for _, sent in ranked_sentences[:top_n]])

def abstractive_summary(text, max_length=130, min_length=30):
    summary = abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# ====== Flask Route ======


@app.route('/summarize', methods=['POST'])
def summarize_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        result = whisper_model.transcribe(file_path)
        transcription = result['text']
        extractive = textrank_summarize(transcription, top_n=5)
        abstractive = abstractive_summary(transcription)
        os.remove(file_path)

        return jsonify({
            'transcription': transcription,
            'extractive_summary': extractive,
            'abstractive_summary': abstractive
        })

    except Exception as e:
        traceback.print_exc()  # 👈 add this line to print the full error
        return jsonify({'error': str(e)}), 500

# ====== Run Server ======
@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
