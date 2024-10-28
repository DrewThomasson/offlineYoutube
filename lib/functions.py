# lib/functions.py

import os
import re
import yt_dlp
import pandas as pd
import numpy as np
import requests
import faiss
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

def initialize_models(whisper_model_size='tiny', device='cpu', compute_type='int8', embedding_model_name='all-MiniLM-L6-v2'):
    """
    Initialize the Whisper and embedding models.
    """
    whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    embedding_model = SentenceTransformer(embedding_model_name)
    return whisper_model, embedding_model

def setup_directories():
    """
    Create necessary directories for storing thumbnails and datasets.
    """
    os.makedirs('thumbnails', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

def extract_video_id_from_link(link):
    """
    Extract YouTube video ID from a link.
    """
    video_id = re.search(r"v=([0-9A-Za-z_-]{11})", link)
    return video_id.group(1) if video_id else None

def get_video_id(youtube_link):
    """
    Get the video ID from a YouTube link.
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_link)
    return match.group(1) if match else None

def download_thumbnail(video_id):
    """
    Download the thumbnail image for a YouTube video.
    """
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    thumbnail_path = f"thumbnails/{video_id}.jpg"
    
    if not os.path.exists(thumbnail_path):
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open(thumbnail_path, 'wb') as f:
                f.write(response.content)
    return thumbnail_path

def extract_transcript(video_url, whisper_model):
    """
    Transcribe the audio of a YouTube video using faster-whisper.
    """
    video_id = get_video_id(video_url)
    print(f"Transcribing {video_id}...")

    try:
        with yt_dlp.YoutubeDL({'format': 'bestaudio'}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            audio_url = info['url']
            video_title = info.get('title', '')
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return [], '', ''

    segments, _ = whisper_model.transcribe(audio_url, vad_filter=True)

    sentences = []
    for segment in segments:
        for sentence in segment.text.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentences.append((sentence, segment.start))
    return sentences, video_id, video_title

def process_videos(video_links, whisper_model):
    """
    Process a list of YouTube videos and extract their transcripts.
    """
    data = []

    for link in video_links:
        sentences, video_id, video_title = extract_transcript(link, whisper_model)
        thumbnail_path = download_thumbnail(video_id)

        for sentence, timestamp in sentences:
            timestamped_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
            data.append({
                'text': sentence,
                'timestamp': timestamp,
                'YouTube_link': link,
                'YouTube_timestamped_link': timestamped_link,
                'thumbnail_path': thumbnail_path,
                'video_title': video_title
            })

    return pd.DataFrame(data)

def save_dataset(data):
    """
    Save the dataset to a CSV file.
    """
    dataset_path = 'datasets/transcript_dataset.csv'
    if os.path.exists(dataset_path):
        existing_data = pd.read_csv(dataset_path)
        data = pd.concat([existing_data, data], ignore_index=True)
    data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

def create_vector_database(embedding_model):
    """
    Create a FAISS vector database from the entire dataset.
    """
    dataset_path = 'datasets/transcript_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please add videos first.")
        return

    data = pd.read_csv(dataset_path)
    data['embedding'] = data['text'].apply(lambda x: embedding_model.encode(x))

    dimension = len(data['embedding'].iloc[0])
    index = faiss.IndexFlatL2(dimension)

    embeddings = np.vstack(data['embedding'].values)
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, 'datasets/vector_index.faiss')
    print("Vector database created and saved.")

def query_vector_database(query, embedding_model, top_k=5):
    """
    Query the FAISS vector database with a search query.
    """
    index = faiss.read_index('datasets/vector_index.faiss')
    data = pd.read_csv('datasets/transcript_dataset.csv')

    query_vector = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = data.loc[indices[0]].copy()
    results['score'] = distances[0]

    # Extract base video link for grouping
    results['video_id'] = results['YouTube_link'].apply(get_video_id)

    # Aggregate most relevant videos by video ID
    video_relevance = (
        results.groupby('video_id')
        .agg(
            relevance=('score', 'mean'),
            thumbnail=('thumbnail_path', 'first'),
            text=('text', 'first'),
            original_link=('YouTube_link', 'first'),
            video_title=('video_title', 'first')
        )
        .sort_values(by='relevance', ascending=True)
        .head(5)
        .reset_index(drop=True)  # Reset index here
    )

    return results[['text', 'YouTube_timestamped_link', 'thumbnail_path', 'score', 'video_title']], video_relevance


def get_video_links(option, input_text):
    """
    Get video links from a playlist or a list of video URLs.
    """
    video_links = []
    if option == 'playlist':
        playlist_url = input_text.strip()
        try:
            with yt_dlp.YoutubeDL({'extract_flat': 'in_playlist'}) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                video_links = [entry['url'] for entry in playlist_info['entries']]
        except Exception as e:
            print(f"Error extracting playlist: {e}")
    elif option == 'videos':
        video_links = input_text.strip().split(',')
    else:
        print("Invalid option.")
    
    return video_links
