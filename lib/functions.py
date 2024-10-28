import os
import re
import yt_dlp
import pandas as pd
import numpy as np
import requests
import faiss
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

# Setup directories
os.makedirs('thumbnails', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

# Initialize models
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_video_id_from_link(link):
    video_id = re.search(r"v=([0-9A-Za-z_-]{11})", link)
    return f"https://www.youtube.com/watch?v={video_id.group(1)}" if video_id else link

def get_video_id(youtube_link):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_link)
    return match.group(1) if match else None

def download_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    thumbnail_path = f"thumbnails/{video_id}.jpg"

    if not os.path.exists(thumbnail_path):
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open(thumbnail_path, 'wb') as f:
                f.write(response.content)
    return thumbnail_path

def extract_transcript(video_url):
    video_id = get_video_id(video_url)
    print(f"Transcribing {video_id}...")

    with yt_dlp.YoutubeDL({'format': 'bestaudio'}) as ydl:
        info = ydl.extract_info(video_url, download=False)
        audio_url = info['url']

    segments, _ = whisper_model.transcribe(audio_url, vad_filter=True)

    sentences = []
    for segment in segments:
        for sentence in segment.text.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentences.append((sentence, segment.start))
    return sentences

def process_videos(video_links):
    data = []

    for link in video_links:
        video_id = get_video_id(link)

        # Extract video metadata (including title)
        with yt_dlp.YoutubeDL({'format': 'bestaudio'}) as ydl:
            info = ydl.extract_info(link, download=False)
            video_title = info.get('title', 'Unknown Title')

        sentences = extract_transcript(link)
        thumbnail_path = download_thumbnail(video_id)

        for sentence, timestamp in sentences:
            data.append({
                'text': sentence,
                'timestamp': timestamp,
                'YouTube_link': link,
                'thumbnail_path': thumbnail_path,
                'video_title': video_title
            })

    return pd.DataFrame(data)


def save_dataset(data):
    dataset_path = 'datasets/transcript_dataset.csv'
    if os.path.exists(dataset_path):
        existing_data = pd.read_csv(dataset_path)
        data = pd.concat([existing_data, data], ignore_index=True)
    data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

def create_vector_database(data):
    data['embedding'] = data['text'].apply(lambda x: embedding_model.encode(x))

    dimension = len(data['embedding'].iloc[0])
    index = faiss.IndexFlatL2(dimension)

    embeddings = np.vstack(data['embedding'].values)
    index.add(embeddings)

    faiss.write_index(index, 'datasets/vector_index.faiss')
    print("Vector database created and saved.")
    return index

def query_vector_database(query, top_k=5):
    index = faiss.read_index('datasets/vector_index.faiss')
    data = pd.read_csv('datasets/transcript_dataset.csv')

    query_vector = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = data.loc[indices[0]].copy()
    results['score'] = distances[0]

    results['video_id'] = results['YouTube_link'].apply(extract_video_id_from_link)

    video_relevance = (
        results.groupby('video_id')
        .agg(
            relevance=('score', 'mean'),
            thumbnail=('thumbnail_path', 'first'),
            text=('text', 'first'),
            original_link=('YouTube_link', 'first')
        )
        .reset_index()
        .sort_values(by='relevance', ascending=True)
        .head(5)
    )

    return results[['text', 'YouTube_link', 'thumbnail_path', 'score']], video_relevance

def get_video_links(playlist_or_links):
    if "youtube.com/playlist" in playlist_or_links:
        with yt_dlp.YoutubeDL({'extract_flat': 'in_playlist'}) as ydl:
            playlist_info = ydl.extract_info(playlist_or_links, download=False)
            video_links = [entry['url'] for entry in playlist_info['entries']]
    else:
        video_links = playlist_or_links.split(',')
    return video_links

