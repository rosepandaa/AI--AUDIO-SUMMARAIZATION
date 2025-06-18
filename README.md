AUDIO SUMMARAIZATION WITH AI

An AI-powered web application that transcribes and summarizes audio recordings using the  models like **Whisper** and **BART**. It helps users extract key insights from long audio files with just a few clicks.

🚀 Features

 **Audio Upload** (MP3, WAV, M4A, FLAC, OGG)
 **Transcription** using OpenAI’s **Whisper**
 **Extractive Summarization** using **TextRank** with TF-IDF
 **Abstractive Summarization** using **BART (facebook/bart-large-cnn)**
 Beautiful interactive **frontend** with dark mode, drag & drop, and audio player
 **Local summary history** feature

 🛠️ Tech Stack

**Frontend**: HTML5, CSS3, JavaScript (Vanilla)
 **Backend**: Flask (Python)
 **AI Models**:
   Whisper (Speech-to-Text)
   TextRank (Extractive Summary via NLTK + NetworkX)
   BART via Hugging Face Transformers (Abstractive Summary)

   📦 Installation

1. Clone the repository:
git clone https://github.com/yourusername/audio-summarizer-bot.git
cd audio-summarizer-bot

2. Install dependencies:
pip install flask openai-whisper nltk transformers torch scikit-learn networkx

3. Download NLTK resources(first run only):
   NOTE: use python ide to download 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

4. Run the Flask app:(NAVIGATE TO THE FOLDER WHERE PYTHON IS INSTALLED)
python app1.py

5. Open your browser and go to:
   http://127.0.0.1:5000/



FILE STRUCTURE
├── app1.py              # Flask backend logic
├── index.html           # Frontend UI
└── uploads/             # Temporary audio storage


