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
import pysrt
import subprocess
import webvtt

# Define the base directory pulled from the app.py
#from ..config import OFFLINE_YOUTUBE_DIR
from offlineyoutube.config import OFFLINE_YOUTUBE_DIR




# lib/functions.py
import os
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
import pysrt
import subprocess
import webvtt
import tempfile
from pathlib import Path

# Define the base directory pulled from the app.py
from config import OFFLINE_YOUTUBE_DIR  # Ensure this path is correct


def initialize_models(whisper_model_size='small', device='cpu', compute_type='int8', embedding_model_name='all-MiniLM-L6-v2'):
    """
    Initialize the Whisper and embedding models.
    """
    try:
        whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        print(f"Initialized WhisperModel with size='{whisper_model_size}', device='{device}', compute_type='{compute_type}'.")
    except Exception as e:
        print(f"Error initializing WhisperModel: {e}")
        raise e

    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Initialized SentenceTransformer with model='{embedding_model_name}'.")
    except Exception as e:
        print(f"Error initializing SentenceTransformer: {e}")
        raise e

    return whisper_model, embedding_model


def setup_directories():
    """
    Create necessary directories for storing thumbnails and datasets within the base directory.
    """
    directories = [
        'thumbnails',
        'datasets',
        'tmp',
        'videos',
        'uploaded_files'
    ]
    for directory in directories:
        path = os.path.join(OFFLINE_YOUTUBE_DIR, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory exists: {path}")


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
    thumbnail_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'thumbnails', f"{video_id}.jpg")
    
    if not os.path.exists(thumbnail_path):
        try:
            response = requests.get(thumbnail_url, stream=True)
            if response.status_code == 200:
                with open(thumbnail_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                print(f"Downloaded thumbnail for video ID {video_id} to {thumbnail_path}.")
            else:
                print(f"Failed to download thumbnail for video ID {video_id}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading thumbnail for video ID {video_id}: {e}")
    else:
        print(f"Thumbnail already exists for video ID {video_id} at {thumbnail_path}.")
    return thumbnail_path


def download_video(video_url, output_dir, keep_video=True, download_audio_only=False, video_quality="720p"):
    """
    Download video or audio to a specified directory, attempt to download subtitles.
    """
    # First, attempt to download subtitles only
    subtitles_available, subtitle_file, video_id, video_title = download_subtitles(video_url, output_dir)
    
    # Define video quality mapping
    quality_mapping = {
        "144p": "bestvideo[height<=144][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "240p": "bestvideo[height<=240][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "360p": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "1080p": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/mp4",
    }

    selected_format = quality_mapping.get(video_quality, "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4")
    print(f"Selected format for download: {selected_format}")

    # Decide whether to download video or audio based on subtitles availability and user preference
    if keep_video:
        # Need to download the video with selected quality
        ydl_opts = {
            'format': selected_format,
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'skip_download': False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=True)
                video_id = info_dict.get('id', '')
                video_title = info_dict.get('title', '')
                # Get the actual filename
                filename = ydl.prepare_filename(info_dict)
                video_file = filename
                print(f"Downloaded video: {video_file}")
        except Exception as e:
            print(f"Error downloading media for video {video_url}: {e}")
            video_file = None
    else:
        # If subtitles are available and not keeping video, we don't need to download anything
        if subtitles_available:
            print("Subtitles found. Proceeding without downloading media.")
            video_file = None
        else:
            # Need to download audio for transcription
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'skip_download': False,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)
                    video_id = info_dict.get('id', '')
                    video_title = info_dict.get('title', '')
                    # Get the actual filename
                    filename = ydl.prepare_filename(info_dict)
                    video_file = filename
                    print(f"Downloaded audio: {video_file}")
            except Exception as e:
                print(f"Error downloading audio for video {video_url}: {e}")
                video_file = None

    return video_file, video_id, video_title, subtitles_available, subtitle_file


def download_subtitles(video_url, output_dir):
    """
    Attempt to download subtitles for a video without downloading the video.
    """
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'outtmpl': os.path.join(output_dir, '%(id)s'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_id = info_dict.get('id', '')
            video_title = info_dict.get('title', '')

            # Check for subtitle files
            subtitle_file = None
            subtitles_available = False
            possible_extensions = ['en.srt', 'en.vtt']
            for ext_sub in possible_extensions:
                possible_subtitle_file = os.path.join(output_dir, f"{video_id}.{ext_sub}")
                if os.path.exists(possible_subtitle_file):
                    subtitle_file = possible_subtitle_file
                    subtitles_available = True
                    print(f"Found subtitle file: {subtitle_file}")
                    break

            # If subtitles are not available, attempt with subprocess
            if not subtitles_available:
                print("Subtitles not found. Attempting to download subtitles using alternative method.")
                cmd = [
                    'yt-dlp', '--skip-download', '--write-sub', '--write-auto-sub',
                    '--sub-lang', 'en', '--output',
                    os.path.join(output_dir, '%(id)s'),
                    video_url
                ]
                subprocess.run(cmd, check=False)
                # Attempt to find the subtitle file
                for ext_sub in possible_extensions:
                    possible_subtitle_file = os.path.join(output_dir, f"{video_id}.{ext_sub}")
                    if os.path.exists(possible_subtitle_file):
                        subtitle_file = possible_subtitle_file
                        subtitles_available = True
                        print(f"Downloaded subtitle file: {subtitle_file}")
                        break

            return subtitles_available, subtitle_file, video_id, video_title

    except Exception as e:
        print(f"Error downloading subtitles for video {video_url}: {e}")
        return False, None, None, None


def extract_audio_from_video(video_file_path):
    """
    Extract audio from a video file using ffmpeg and save it to a temporary file.
    Returns the path to the extracted audio file.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        audio_file_path = os.path.join(temp_dir, "extracted_audio.wav")
        cmd = [
            'ffmpeg',
            '-i', video_file_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little endian
            '-ar', '16000',  # 16kHz
            '-ac', '1',  # Mono
            audio_file_path,
            '-y'  # Overwrite without asking
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Extracted audio to {audio_file_path}")
        return audio_file_path
    except Exception as e:
        print(f"Error extracting audio from {video_file_path}: {e}")
        return None


def convert_to_mp4(input_file, output_dir):
    """
    Convert any video or audio file to MP4 format using ffmpeg.
    Returns the path to the converted MP4 file.
    """
    try:
        input_path = Path(input_file)
        output_path = Path(output_dir) / (input_path.stem + ".mp4")
        if input_path.suffix.lower() != '.mp4':
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-b:a', '192k',
                '-y',  # Overwrite without asking
                str(output_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"Converted {input_file} to {output_path}")
            return str(output_path)
        else:
            # If already mp4, just return the original path
            print(f"File {input_file} is already in MP4 format.")
            return str(input_file)
    except Exception as e:
        print(f"Error converting {input_file} to MP4: {e}")
        return None


def extract_transcript(audio_file, whisper_model, subtitles_available=False, subtitle_file=None):
    """
    Transcribe the audio file using faster-whisper or read subtitles.
    """
    if subtitles_available and subtitle_file:
        # Read subtitles file
        sentences = extract_transcript_from_subtitles(subtitle_file)
    elif audio_file:
        # Transcribe using Whisper
        print("Using Whisper to transcribe audio.")
        sentences = []
        try:
            # Reduced beam size and no VAD filter to stabilize
            segments, _ = whisper_model.transcribe(audio_file, vad_filter=False, beam_size=5)
            for segment in segments:
                for sentence in segment.text.split('.'):
                    sentence = sentence.strip()
                    if sentence:
                        sentences.append((sentence, segment.start))
            print(f"Transcription completed for {audio_file}.")
        except Exception as e:
            print(f"Error during transcription: {e}")
            sentences = []
    else:
        print("No subtitles or audio file available for transcription.")
        sentences = []
    return sentences


def extract_transcript_from_subtitles(subtitle_file):
    """
    Extract transcript from subtitles file (.srt or .vtt format).
    """
    sentences = []
    try:
        if subtitle_file.endswith('.srt'):
            subs = pysrt.open(subtitle_file)
            for sub in subs:
                text = sub.text.strip().replace('\n', ' ')
                start = sub.start.ordinal / 1000.0  # Convert milliseconds to seconds
                if text:
                    sentences.append((text, start))
        elif subtitle_file.endswith('.vtt'):
            subs = webvtt.read(subtitle_file)
            for caption in subs:
                text = caption.text.strip().replace('\n', ' ')
                start = caption.start_in_seconds
                if text:
                    sentences.append((text, start))
        else:
            print(f"Unsupported subtitle format for file: {subtitle_file}")
    except Exception as e:
        print(f"Error reading subtitles file {subtitle_file}: {e}")
    return sentences

def query_vector_database(query, embedding_model, top_k=5):
    """
    Query the FAISS vector database with a search query.
    """
    index_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'vector_index.faiss')
    dataset_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'transcript_dataset.csv')

    if not os.path.exists(index_path):
        raise FileNotFoundError("Vector index not found. Please add videos first.")

    index = faiss.read_index(index_path)
    data = pd.read_csv(dataset_path)
    if 'video_id' not in data.columns:
        data['video_id'] = data['YouTube_link'].apply(get_video_id)

    query_vector = embedding_model.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = data.iloc[indices[0]].copy()
    results['score'] = distances[0]

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


def process_videos(video_links, uploaded_files_paths, whisper_model, embedding_model, keep_videos=False, video_quality="720p"):
    """
    Process each YouTube video and uploaded files one by one, updating the dataset and vector database after each.
    """
    # Paths for dataset and index
    video_titles = set()  # Use a set to store unique video titles
    dataset_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'transcript_dataset.csv')
    index_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'vector_index.faiss')

    # Decide on video directory
    if keep_videos:
        video_dir = os.path.join(OFFLINE_YOUTUBE_DIR, 'videos')
    else:
        video_dir = os.path.join(OFFLINE_YOUTUBE_DIR, 'tmp')

    os.makedirs(video_dir, exist_ok=True)
    print(f"Using video directory: {video_dir}")

    # Load existing dataset if it exists
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        if 'video_id' not in data.columns:
            data['video_id'] = data['YouTube_link'].apply(get_video_id)
            data.to_csv(dataset_path, index=False)
        existing_video_ids = set(data['video_id'].unique())
        print(f"Loaded existing dataset with {len(existing_video_ids)} videos.")
    else:
        data = pd.DataFrame()
        existing_video_ids = set()
        print("No existing dataset found. Starting fresh.")

    # Load existing index if it exists
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"Loaded existing FAISS index from {index_path}.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            index = None
    else:
        index = None
        print("No existing FAISS index found. A new index will be created.")

    # Process video links
    if video_links:
        for idx, link in enumerate(tqdm(video_links, desc="Processing Videos", unit="video")):
            video_id = get_video_id(link)
            if video_id in existing_video_ids:
                print(f"Video {video_id} already processed. Skipping.")
                continue  # Skip already processed videos

            print(f"\nProcessing video {idx + 1}/{len(video_links)}: {link}")
            # Determine if we need to download audio-only
            download_audio_only = not keep_videos

            # Download video or audio and subtitles with selected video quality
            video_file, video_id, video_title, subtitles_available, subtitle_file = download_video(
                link, video_dir, keep_video=keep_videos, download_audio_only=download_audio_only, video_quality=video_quality
            )

            if not subtitles_available and not video_file:
                print(f"Cannot process video {video_id} because neither subtitles nor audio/video are available.")
                continue

            # Transcribe audio or read subtitles
            print(f"Extracting transcript for video ID {video_id}...")
            if subtitles_available:
                print("Subtitles found. Using subtitles for transcript.")
            else:
                print("Subtitles not found. Using Whisper to transcribe audio.")

            sentences = extract_transcript(video_file, whisper_model, subtitles_available, subtitle_file)
            if not sentences:
                print(f"No transcript available for video {video_id}. Skipping.")
                continue
            thumbnail_path = download_thumbnail(video_id)

            new_data = []
            embeddings = []
            for sentence, timestamp in sentences:
                timestamped_link = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
                local_video_path = os.path.abspath(video_file) if keep_videos and video_file else ''
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
                video_titles.add(video_title)
                # Encode the sentence to get embedding
                embedding = embedding_model.encode(sentence).astype('float32')
                embeddings.append(embedding)

            # Convert new_data to DataFrame
            new_data_df = pd.DataFrame(new_data)

            # Append new data to dataset
            data = pd.concat([data, new_data_df], ignore_index=True)
            # Save updated dataset
            data.to_csv(dataset_path, index=False)
            print(f"Updated dataset with {len(new_data_df)} new entries.")

            # Update the FAISS index
            if embeddings:
                embeddings = np.vstack(embeddings)
                dimension = embeddings.shape[1]
                if index is None:
                    # Create new index
                    index = faiss.IndexFlatL2(dimension)
                    print(f"Created new FAISS index with dimension {dimension}.")
                index.add(embeddings)
                # Save the updated index
                faiss.write_index(index, index_path)
                print(f"Updated FAISS index with {len(embeddings)} new embeddings.")

            # Delete the audio/video file after processing if not keeping videos
            if not keep_videos and video_file and os.path.exists(video_file):
                os.remove(video_file)
                print(f"Deleted temporary video file: {video_file}")
            if subtitles_available and subtitle_file and os.path.exists(subtitle_file):
                os.remove(subtitle_file)
                print(f"Deleted temporary subtitle file: {subtitle_file}")

    # Process uploaded files
    if uploaded_files_paths:
        for idx, file_path in enumerate(tqdm(uploaded_files_paths, desc="Processing Uploaded Files", unit="file")):
            file_extension = os.path.splitext(file_path)[1].lower()
            is_video = file_extension in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
            is_audio = file_extension in ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a']

            if not (is_video or is_audio):
                print(f"Unsupported file type for file {file_path}. Skipping.")
                continue

            video_id = os.path.splitext(os.path.basename(file_path))[0]
            video_title = video_id
            link = ''
            thumbnail_path = ''
            print(f"\nProcessing uploaded file {idx + 1}/{len(uploaded_files_paths)}: {file_path}")

            # Convert to MP4 if not already
            converted_mp4 = convert_to_mp4(file_path, os.path.join(OFFLINE_YOUTUBE_DIR, 'uploaded_files'))
            if not converted_mp4:
                print(f"Failed to convert {file_path} to MP4. Skipping.")
                continue

            # Extract audio from the converted MP4
            audio_file_path = extract_audio_from_video(converted_mp4)
            if not audio_file_path:
                print(f"Failed to extract audio from {converted_mp4}. Skipping.")
                continue

            # Transcribe using Whisper
            print(f"Transcribing uploaded file {video_id}...")
            sentences = extract_transcript(audio_file_path, whisper_model, subtitles_available=False, subtitle_file=None)
            if not sentences:
                print(f"No transcript available for file {video_id}. Skipping.")
                if os.path.exists(audio_file_path):
                    shutil.rmtree(os.path.dirname(audio_file_path))
                continue

            new_data = []
            embeddings = []
            for sentence, timestamp in sentences:
                timestamped_link = ''  # No YouTube link for uploaded files
                local_video_path = os.path.abspath(converted_mp4)  # Always keep uploaded files locally
                new_data.append({
                    'video_id': video_id,
                    'text': sentence,
                    'timestamp': timestamp,
                    'YouTube_link': link,
                    'YouTube_timestamped_link': timestamped_link,
                    'thumbnail_path': thumbnail_path,  # No thumbnail for uploaded files
                    'video_title': video_title,
                    'local_video_path': local_video_path
                })
                video_titles.add(video_title)
                # Encode the sentence to get embedding
                embedding = embedding_model.encode(sentence).astype('float32')
                embeddings.append(embedding)

            # Convert new_data to DataFrame
            new_data_df = pd.DataFrame(new_data)

            # Append new data to dataset
            data = pd.concat([data, new_data_df], ignore_index=True)
            # Save updated dataset
            data.to_csv(dataset_path, index=False)
            print(f"Updated dataset with {len(new_data_df)} new entries from uploaded files.")

            # Update the FAISS index
            if embeddings:
                embeddings = np.vstack(embeddings)
                dimension = embeddings.shape[1]
                if index is None:
                    # Create new index
                    index = faiss.IndexFlatL2(dimension)
                    print(f"Created new FAISS index with dimension {dimension}.")
                index.add(embeddings)
                # Save the updated index
                faiss.write_index(index, index_path)
                print(f"Updated FAISS index with {len(embeddings)} new embeddings.")

            # Delete the extracted audio file after processing
            if os.path.exists(audio_file_path):
                shutil.rmtree(os.path.dirname(audio_file_path))
                print(f"Deleted temporary audio file directory: {os.path.dirname(audio_file_path)}")

        return results[['text', 'YouTube_timestamped_link', 'thumbnail_path', 'score', 'video_title', 'local_video_path', 'timestamp']], video_relevance


def is_channel_url(url):
    """
    Check if a URL is a YouTube channel URL.
    """
    return any(x in url for x in ['/channel/', '/c/', '/user/'])


def get_video_links(input_text, process_channel=False):
    """
    Get video links from a list of input links, automatically detecting playlists, channels, and individual videos.
    """
    video_links = []
    if not input_text.strip():
        return video_links
    links = [link.strip() for link in input_text.strip().split(',') if link.strip()]
    for link in links:
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': 'in_playlist',
            }
            if is_channel_url(link):
                if not process_channel:
                    print(f"Channel URL detected: {link}")
                    print("Process Channel option is not enabled. Skipping channel.")
                    continue
                else:
                    # For channels, get all videos
                    ydl_opts['playlistend'] = None
            else:
                # For non-channels, get all videos in playlists
                ydl_opts['playlistend'] = None
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(link, download=False)
                if '_type' in info and info['_type'] == 'playlist':
                    # It's a playlist or a channel
                    entries = info.get('entries', [])
                    for entry in entries:
                        video_id = entry.get('id')
                        if video_id:
                            video_link = f"https://www.youtube.com/watch?v={video_id}"
                            video_links.append(video_link)
                elif 'id' in info:
                    # It's a single video
                    video_id = info['id']
                    video_link = f"https://www.youtube.com/watch?v={video_id}"
                    video_links.append(video_link)
                else:
                    print(f"Unknown link type, skipped: {link}")
        except Exception as e:
            print(f"Error processing link {link}: {e}")
    return video_links
