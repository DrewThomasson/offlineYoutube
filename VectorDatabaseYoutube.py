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


# Helper function to extract YouTube video ID
def get_video_id(youtube_link):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_link)
    return match.group(1) if match else None

# Download thumbnail for offline use
def download_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    thumbnail_path = f"thumbnails/{video_id}.jpg"
    
    if not os.path.exists(thumbnail_path):
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open(thumbnail_path, 'wb') as f:
                f.write(response.content)
    return thumbnail_path

# Transcribe audio with faster-whisper
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

# Process videos into a dataset
def process_videos(video_links):
    data = []

    for link in video_links:
        video_id = get_video_id(link)
        sentences = extract_transcript(link)
        thumbnail_path = download_thumbnail(video_id)

        for sentence, timestamp in sentences:
            data.append({
                'text': sentence,
                'timestamp': timestamp,
                'YouTube_link': link,
                'thumbnail_path': thumbnail_path
            })

    return pd.DataFrame(data)

# Save dataset to CSV
def save_dataset(data):
    dataset_path = 'datasets/transcript_dataset.csv'
    if os.path.exists(dataset_path):
        existing_data = pd.read_csv(dataset_path)
        data = pd.concat([existing_data, data], ignore_index=True)
    data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

# Create a vector database using FAISS
def create_vector_database(data):
    data['embedding'] = data['text'].apply(lambda x: embedding_model.encode(x))

    dimension = len(data['embedding'].iloc[0])
    index = faiss.IndexFlatL2(dimension)

    embeddings = np.vstack(data['embedding'].values)
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, 'datasets/vector_index.faiss')
    print("Vector database created and saved.")
    return index

# Query the vector database
def query_vector_database(query, top_k=5):
    index = faiss.read_index('datasets/vector_index.faiss')
    data = pd.read_csv('datasets/transcript_dataset.csv')

    query_vector = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = data.loc[indices[0]].copy()  # Avoid SettingWithCopyWarning
    results['score'] = distances[0]

    # Extract base video link for grouping
    results['video_id'] = results['YouTube_link'].apply(extract_video_id_from_link)

    # Aggregate most relevant videos by video ID
    video_relevance = (
        results.groupby('video_id')
        .agg(
            relevance=('score', 'mean'),  # Average relevance for each video
            thumbnail=('thumbnail_path', 'first'),  # Use the first thumbnail
            text=('text', 'first'),  # Use the first text snippet
            original_link=('YouTube_link', 'first')  # Use the first timestamped link
        )
        .reset_index()
        .sort_values(by='relevance', ascending=True)  # Sort by relevance (lower is better)
        .head(5)  # Limit to top 5 videos
    )

    return results[['text', 'YouTube_link', 'thumbnail_path', 'score']], video_relevance


# Main function to handle video input and queries
def main():
    if not os.path.exists('datasets/transcript_dataset.csv'):
        print("No database found. Please add videos to create the initial database.")
        video_links = get_video_links()
        data = process_videos(video_links)
        save_dataset(data)
        create_vector_database(data)
    else:
        print("1: Add more videos\n2: Query the existing database")
        option = input("Select an option: ").strip()

        if option == '1':
            video_links = get_video_links()
            data = process_videos(video_links)
            save_dataset(data)
            create_vector_database(data)
        elif option == '2':
            query_loop()
        else:
            print("Invalid option.")

def get_video_links():
    print("1: Provide a playlist link\n2: Provide a list of video links")
    option = input("Select an option: ").strip()

    if option == '1':
        playlist_url = input("Enter YouTube playlist URL: ").strip()
        with yt_dlp.YoutubeDL({'extract_flat': 'in_playlist'}) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            video_links = [entry['url'] for entry in playlist_info['entries']]
    elif option == '2':
        video_links = input("Enter YouTube video links (comma-separated): ").strip().split(',')
    else:
        print("Invalid option.")
        return []

    return video_links

def query_loop():
    while True:
        query = input("Enter your search query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        results, top_videos = query_vector_database(query)

        # Print detailed results for each text entry
        print("\nDetailed Results:\n")
        for _, row in results.iterrows():
            print(f"Text: {row['text']}")
            print(f"Link: {row['YouTube_link']}")
            print(f"Thumbnail: {row['thumbnail_path']}")
            print(f"Score: {row['score']:.4f}\n")

        # Print top-ranked videos based on relevance
        print("\nTop Relevant Videos:\n")
        for idx, row in top_videos.iterrows():
            print(f"Rank {idx + 1}:")
            print(f"Relevance Score: {row['relevance']:.4f}")
            print(f"Video Link: {row['original_link']}")
            print(f"Thumbnail: {row['thumbnail']}")
            print(f"Example Text: {row['text']}\n")

# Run the application
if __name__ == "__main__":
    main()