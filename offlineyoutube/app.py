# app.py

import os
import sys
sys.path.append(os.path.dirname(__file__))  # Add this line here


import multiprocessing
import gradio as gr
import argparse
import pandas as pd  # Added to handle NaN values
from lib.functions import (
    initialize_models, setup_directories, process_videos,
    query_vector_database, get_video_links
)
# Define the base directory
from .config import OFFLINE_YOUTUBE_DIR





# Initialize models at the top level
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setup_directories()
whisper_model, embedding_model = initialize_models()

# app.py

import os
import sys
import multiprocessing
import shutil  # Added import
import gradio as gr
import argparse
import pandas as pd  # Added to handle NaN values
from lib.functions import (
    initialize_models, setup_directories, process_videos,
    query_vector_database, get_video_links
)
# Define the base directory
from config import OFFLINE_YOUTUBE_DIR  # Ensure this path is correct

# Initialize models at the top level
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setup_directories()
whisper_model, embedding_model = initialize_models()


def add_videos_interface(input_text, uploaded_files, process_channel, keep_videos, video_quality):
    """
    Interface function for adding videos to the database.
    """
    # Use models directly from the outer scope
    video_links = get_video_links(input_text, process_channel)
    uploaded_files_paths = []
    if uploaded_files:
        uploaded_files_dir = os.path.join(OFFLINE_YOUTUBE_DIR, 'uploaded_files')
        os.makedirs(uploaded_files_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            # Gradio v3.36.1 provides uploaded_file as a _TemporaryFileWrapper object
            # 'uploaded_file.name' contains the path to the temporary file
            try:
                # Extract the original filename from the temporary file path
                original_filename = os.path.basename(uploaded_file.name)
                file_path = os.path.join(uploaded_files_dir, original_filename)
                
                # Copy the file from the temporary location to the desired directory
                shutil.copy(uploaded_file.name, file_path)
                
                # Verify that the file has been copied correctly and is not empty
                if os.path.getsize(file_path) == 0:
                    print(f"Uploaded file {original_filename} is empty. Skipping.")
                    continue
                uploaded_files_paths.append(file_path)
                print(f"Saved uploaded file {original_filename} to {file_path} ({os.path.getsize(file_path)} bytes)")
            except Exception as e:
                print(f"Error saving uploaded file {original_filename}: {e}")
    if not video_links and not uploaded_files_paths:
        return "No valid video links or files provided."
    # Process videos and uploaded files with selected video quality
    data, video_titles = process_videos(
        video_links, uploaded_files_paths, whisper_model, embedding_model, keep_videos=keep_videos, video_quality=video_quality
    )
    
    # Prepare a message with the video titles
    if video_titles:
        titles_message = "\n".join(f"- {title}" for title in video_titles)
        return f"Videos processed and database updated.\nAdded Videos:\n{titles_message}"
    else:
        return "No new videos were added to the database."


def search_interface(query_text, top_k):
    """
    Interface function for searching the database.
    """
    index_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'vector_index.faiss')
    dataset_path = os.path.join(OFFLINE_YOUTUBE_DIR, 'datasets', 'transcript_dataset.csv')
    
    if not os.path.exists(index_path):
        return "No database found. Please add videos first.", None
    try:
        results, top_videos = query_vector_database(query_text, embedding_model, top_k=top_k)
    except Exception as e:
        return f"Error: {e}", None

    # Prepare top videos
    top_videos_html = "<h1>Top Relevant Videos:</h1>"
    for idx, row in top_videos.iterrows():
        rank = idx + 1  # Since idx is now sequential
        # Check if local video exists
        local_video_path = row['local_video_path']
        if isinstance(local_video_path, str) and local_video_path and not pd.isnull(local_video_path):
            local_video_exists = os.path.exists(local_video_path)
        else:
            local_video_exists = False
        local_video_player = ''
        if local_video_exists:
            # Replace backslashes with forward slashes for compatibility
            local_video_url = 'file/' + local_video_path.replace("\\", "/")
            local_video_player = f"""
            <details>
                <summary>Show Local Video</summary>
                <video width='320' height='240' controls>
                    <source src='{local_video_url}' type='video/mp4'>
                    Your browser does not support the video tag.
                </video>
            </details>
            """
        top_videos_html += f"""
        <div style='margin-bottom:20px;'>
            <h4>Rank {rank}</h4>
            <img src='file/{row['thumbnail']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Relevance Score:</strong> {row['relevance']:.4f}</p>
            <p><strong>Example Text:</strong> {row['text']}</p>
            <p><a href='{row['original_link']}' target='_blank'>Watch on YouTube</a></p>
            {local_video_player}
            <div style='clear:both;'></div>
        </div>
        """

    # Prepare detailed results
    detailed_html = "<h1>Detailed Results:</h1>"
    for _, row in results.iterrows():
        # Check if local video exists
        local_video_path = row['local_video_path']
        if isinstance(local_video_path, str) and local_video_path and not pd.isnull(local_video_path):
            local_video_exists = os.path.exists(local_video_path)
        else:
            local_video_exists = False
        local_video_player = ''
        if local_video_exists:
            # Replace backslashes with forward slashes for compatibility
            local_video_url = 'file/' + local_video_path.replace("\\", "/")
            timestamp = int(row['timestamp'])
            local_video_player = f"""
            <details>
                <summary>Show Local Video at Timestamp</summary>
                <video width='320' height='240' controls>
                    <source src='{local_video_url}#t={timestamp}' type='video/mp4'>
                    Your browser does not support the video tag.
                </video>
            </details>
            """
        detailed_html += f"""
        <div style='margin-bottom:20px;'>
            <img src='file/{row['thumbnail_path']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Text:</strong> {row['text']}</p>
            <p><strong>Score:</strong> {row['score']:.4f}</p>
            <p><a href='{row['YouTube_timestamped_link']}' target='_blank'>Watch on YouTube at Timestamp</a></p>
            {local_video_player}
            <div style='clear:both;'></div>
        </div>
        """
    return top_videos_html, detailed_html


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Search Application",
        epilog="""
Examples:
  # Add videos from a playlist and keep videos locally
  python app.py add --input "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" --keep_videos
  
  # Add specific videos without keeping videos locally
  python app.py add --input "https://www.youtube.com/watch?v=VIDEO_ID1,https://www.youtube.com/watch?v=VIDEO_ID2"
  
  # Search the database with a query
  python app.py search --query "Your search query" --top_k 5
  
  # Run the Gradio web interface
  python app.py ui
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command')

    # Add videos command
    parser_add = subparsers.add_parser('add', help='Add videos to the database')
    parser_add.add_argument('--input', required=True, help='Playlist URL or comma-separated video URLs')
    parser_add.add_argument('--process_channel', action='store_true', help='Process entire channel when a channel URL is provided')
    parser_add.add_argument('--keep_videos', action='store_true', help='Keep videos stored locally')

    # Search command
    parser_search = subparsers.add_parser('search', help='Search the video database')
    parser_search.add_argument('--query', required=True, help='Search query')
    parser_search.add_argument('--top_k', type=int, default=5, help='Number of results to return')

    # Run Gradio interface
    parser_ui = subparsers.add_parser('ui', help='Run the Gradio web interface')

    args = parser.parse_args()

    if args.command == 'add':
        # For CLI, we will use the default video quality of 720p
        default_video_quality = "720p"
        status = add_videos_interface(args.input, [], args.process_channel, args.keep_videos, default_video_quality)
        print(status)

    elif args.command == 'search':
        top_videos_html, detailed_results = search_interface(args.query, args.top_k)
        if isinstance(top_videos_html, str):
            print(top_videos_html)
        else:
            # Extract data from HTML for console output
            from bs4 import BeautifulSoup

            # Extract top videos
            soup = BeautifulSoup(top_videos_html, 'html.parser')
            print("Top Relevant Videos:\n")
            for idx, div in enumerate(soup.find_all('div')):
                rank = div.find('h4').text
                title = div.find('p', text=lambda t: t and 'Title:' in t).text
                relevance = div.find('p', text=lambda t: t and 'Relevance Score:' in t).text
                example_text = div.find('p', text=lambda t: t and 'Example Text:' in t).text
                link = div.find('a')['href']
                print(f"{rank}\n{title}\n{relevance}\n{example_text}\nLink: {link}\n")

            # Extract detailed results
            soup = BeautifulSoup(detailed_results, 'html.parser')
            print("Detailed Results:\n")
            for div in soup.find_all('div'):
                title = div.find('p', text=lambda t: t and 'Title:' in t).text
                text = div.find('p', text=lambda t: t and 'Text:' in t).text
                score = div.find('p', text=lambda t: t and 'Score:' in t).text
                link = div.find('a')['href']
                print(f"{title}\n{score}\n{text}\nLink: {link}\n")

    else:
        # Run Gradio interface if no command is provided or 'ui' command is used
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎥 YouTube Video Search Application")

            with gr.Tab("Add Videos"):
                gr.Markdown("### Add videos to the database")
                input_text = gr.Textbox(lines=2, placeholder="Enter playlist, channel, and/or video URLs (comma-separated)")
                process_channel = gr.Checkbox(label="Process entire channel when a channel URL is provided", value=False)
                keep_videos = gr.Checkbox(label="Keep videos stored locally", value=True)
                video_quality = gr.Dropdown(
                    label="Select Video Quality",
                    choices=["144p", "240p", "360p", "480p", "720p", "1080p"],
                    value="720p",
                    info="Choose the desired video quality for downloads."
                )
                file_upload = gr.File(label="Upload your own video/audio files", file_count="multiple", type="file")
                add_button = gr.Button("Add Videos")
                add_output = gr.Textbox(label="Status")
                add_button.click(
                    add_videos_interface,
                    inputs=[input_text, file_upload, process_channel, keep_videos, video_quality],
                    outputs=add_output
                )

            with gr.Tab("Search"):
                gr.Markdown("### Search the video database")
                query_text = gr.Textbox(lines=1, placeholder="Enter your search query")
                top_k = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
                search_button = gr.Button("Search")
                top_video_results = gr.HTML()
                detailed_results = gr.HTML()
                search_button.click(
                    search_interface,
                    inputs=[query_text, top_k],
                    outputs=[top_video_results, detailed_results]
                )

        demo.launch()


if __name__ == "__main__":
    # Fix for multiprocessing in PyInstaller
    multiprocessing.freeze_support()

    # Ensure set_start_method is only set once
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
