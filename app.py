import streamlit as st
import numpy as np
import json
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import chromadb
import zipfile
import os

# Loading the environment variables
api_key = os.environ.get("api_key")

# Initialize AssemblyAI API key from environment variables.
aai.settings.api_key = api_key

# Unzip the db folder.
db_zip_path = os.path.join(".", "db.zip") 
db_extract_path = os.path.join(".", "db_extracted")

if os.path.exists(db_zip_path):
    os.makedirs(db_extract_path, exist_ok=True)  # create the folder if it does not exist.
    with zipfile.ZipFile(db_zip_path, 'r') as zip_ref:
        zip_ref.extractall(db_extract_path)
    print(f"db.zip extracted to: {db_extract_path}")

# ChromaDB Setup configurations
client = chromadb.PersistentClient(path=db_extract_path)
collection = client.get_or_create_collection(name="subtitle_chunks")

def transcribe_audio(audio_file):
    """Transcribes audio from the given file using AssemblyAI."""
    if audio_file is None:
        return "Please upload an audio file.", None

    config = aai.TranscriptionConfig(language_code="en")
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_file)

    return transcript.text, transcript.text

def format_results_as_json(results):
    """Formats subtitle search results into a JSON string."""
    formatted_results = []
    if results and results["metadatas"] and results["metadatas"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            subtitle_name = metadata["subtitle_name"]
            subtitle_id = metadata["subtitle_id"]
            url = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}"
            formatted_results.append({
                "Result": i + 1,
                "Subtitle Name": subtitle_name.upper(),
                "URL": url,
            })
        return json.dumps(formatted_results, indent=4)
    else:
        return json.dumps([{"Result": "No results found"}], indent=4)

def retrieve_and_display_results(query):
    """Retrieves and displays subtitle search results based on the given query."""
    if not query:
        return "No transcription text available for search."

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5, include=["metadatas"])

    return format_results_as_json(results)

def clear_all():
    """Clears the transcribed text and search results."""
    st.session_state.transcribed_text = ""
    st.session_state.search_results = ""

def main():
    st.title("🎵 Shazam Clone : Audio Transcription & Subtitle Search")

    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = ""

    audio_input = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Transcribe"):
            if audio_input:
                transcribed_text, _ = transcribe_audio(audio_input)
                st.session_state.transcribed_text = transcribed_text
            else:
                st.warning("Please upload an audio file.")

    with col2:
        if st.button("Search Subtitles"):
            if st.session_state.transcribed_text:
                st.session_state.search_results = retrieve_and_display_results(st.session_state.transcribed_text)
            else:
                st.warning("Please transcribe audio first.")

    st.text_area("Transcribed Text", value=st.session_state.transcribed_text, height=150)
    st.text_area("Subtitle Search Results", value=st.session_state.search_results, height=300)

    if st.button("Clear"):
        clear_all()
        st.experimental_rerun()

if __name__== "__main__":
    main()
