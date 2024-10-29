# lib/functions.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import yt_dlp
import pandas as pd
import numpy as np
import requests
import faiss
import shutil
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
    os.makedirs('tmp', exist_ok=True)  # Temporary directory for downloaded videos
    os.makedirs('videos', exist_ok=True)  # Permanent directory for videos if needed

def extract_video_id_from_link(link):
    """
    Extract YouTube video ID from a link.
    """
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", link)
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

def download_video(video_url, output_dir):
    """
    Download video to a specified directory.
    """
    ydl_opts = {
        #'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        # This is setting the highest video quality allowed being 720 
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_id = info_dict.get('id', '')
            video_title = info_dict.get('title', '')
            ext = 'mp4'
            filename = os.path.join(output_dir, f"{video_id}.{ext}")
            return filename, video_id, video_title
    except Exception as e:
        print(f"Error downloading video {video_url}: {e}")
        return None, None, None

def extract_transcript(audio_file, whisper_model):
    """
    Transcribe the audio file using faster-whisper.
    """
    segments, _ = whisper_model.transcribe(audio_file, vad_filter=True)
    
    sentences = []
    for segment in segments:
        for sentence in segment.text.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentences.append((sentence, segment.start))
    return sentences

def process_videos(video_links, whisper_model, embedding_model, keep_videos=False):
    """
    Process each YouTube video one by one, updating the dataset and vector database after each.
    """
    # Paths for dataset and index
    dataset_path = 'datasets/transcript_dataset.csv'
    index_path = 'datasets/vector_index.faiss'

    # Decide on video directory
    if keep_videos:
        video_dir = 'videos'
    else:
        video_dir = 'tmp'

    os.makedirs(video_dir, exist_ok=True)

    # Load existing dataset if it exists
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        if 'video_id' not in data.columns:
            data['video_id'] = data['YouTube_link'].apply(get_video_id)
            data.to_csv(dataset_path, index=False)
        existing_video_ids = set(data['video_id'].unique())
    else:
        data = pd.DataFrame()
        existing_video_ids = set()

    # Load existing index if it exists
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = None

    for idx, link in enumerate(tqdm(video_links, desc="Processing Videos", unit="video")):
        video_id = get_video_id(link)
        if video_id in existing_video_ids:
            print(f"Video {video_id} already processed. Skipping.")
            continue  # Skip already processed videos

        print(f"\nProcessing video {idx + 1}/{len(video_links)}: {link}")
        # Download video
        video_file, video_id, video_title = download_video(link, video_dir)
        if not video_file:
            continue  # Skip if download failed

        # Transcribe audio
        print(f"Transcribing video ID {video_id}...")
        sentences = extract_transcript(video_file, whisper_model)
        thumbnail_path = download_thumbnail(video_id)

        new_data = []
        embeddings = []
        for sentence, timestamp in sentences:
            timestamped_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
            local_video_path = os.path.abspath(video_file) if keep_videos else ''
            new_data.append({
                'video_id': video_id,
                'text': sentence,
                'timestamp': timestamp,
                'YouTube_link': link,
                'YouTube_timestamped_link': timestamped_link,
                'thumbnail_path': thumbnail_path,
                'video_title': video_title,
                'local_video_path': local_video_path
            })
            # Encode the sentence to get embedding
            embedding = embedding_model.encode(sentence).astype('float32')
            embeddings.append(embedding)

        # Convert new_data to DataFrame
        new_data_df = pd.DataFrame(new_data)

        # Append new data to dataset
        data = pd.concat([data, new_data_df], ignore_index=True)
        # Save updated dataset
        data.to_csv(dataset_path, index=False)
        # Update existing_video_ids
        existing_video_ids.add(video_id)

        # Update the FAISS index
        embeddings = np.vstack(embeddings)
        dimension = embeddings.shape[1]
        if index is None:
            # Create new index
            index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        # Save the updated index
        faiss.write_index(index, index_path)

        # Delete the video file after processing if not keeping videos
        if not keep_videos:
            os.remove(video_file)

    # Delete the tmp directory and all its contents if not keeping videos
    if not keep_videos and os.path.exists('tmp'):
        shutil.rmtree('tmp')
    print("All videos have been processed and added to the database.")
    return data

def query_vector_database(query, embedding_model, top_k=5):
    """
    Query the FAISS vector database with a search query.
    """
    index = faiss.read_index('datasets/vector_index.faiss')
    data = pd.read_csv('datasets/transcript_dataset.csv')
    if 'video_id' not in data.columns:
        data['video_id'] = data['YouTube_link'].apply(get_video_id)

    query_vector = embedding_model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = data.iloc[indices[0]].copy()
    results['score'] = distances[0]

    # Extract base video link for grouping (already have 'video_id' column)
    # results['video_id'] = results['YouTube_link'].apply(get_video_id)

    # Aggregate most relevant videos by video ID
    video_relevance = (
        results.groupby('video_id')
        .agg(
            relevance=('score', 'mean'),
            thumbnail=('thumbnail_path', 'first'),
            text=('text', 'first'),
            original_link=('YouTube_link', 'first'),
            video_title=('video_title', 'first'),
            local_video_path=('local_video_path', 'first')
        )
        .sort_values(by='relevance', ascending=True)
        .head(5)
        .reset_index(drop=True)
    )

    return results[['text', 'YouTube_timestamped_link', 'thumbnail_path', 'score', 'video_title', 'local_video_path', 'timestamp']], video_relevance

def get_video_links(option, input_text):
    """
    Get video links from a playlist or a list of video URLs.
    """
    video_links = []
    if option == 'playlist':
        playlist_url = input_text.strip()
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': 'in_playlist'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                video_links = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in playlist_info['entries']]
        except Exception as e:
            print(f"Error extracting playlist: {e}")
    elif option == 'videos':
        video_links = [link.strip() for link in input_text.strip().split(',')]
    else:
        print("Invalid option.")
    
    return video_links
