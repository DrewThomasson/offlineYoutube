# app.py

import gradio as gr
import argparse
from lib.functions import (
    initialize_models, setup_directories, process_videos,
    save_dataset, create_vector_database, query_vector_database,
    get_video_links
)
import os
import sys

# Initialize models
setup_directories()
whisper_model, embedding_model = initialize_models()

def add_videos_interface(option, input_text):
    """
    Interface function for adding videos to the database.
    """
    video_links = get_video_links(option, input_text)
    if not video_links:
        return "No valid video links provided."
    data = process_videos(video_links, whisper_model)
    save_dataset(data)
    create_vector_database(embedding_model)
    return "Videos processed and database updated."

def add_videos_interface(option, input_text):
    """
    Interface function for adding videos to the database.
    """
    video_links = get_video_links(option, input_text)
    if not video_links:
        return "No valid video links provided."
    data = process_videos(video_links, whisper_model)
    save_dataset(data)
    create_vector_database(embedding_model)
    return "Videos processed and database updated."

def search_interface(query_text, top_k):
    """
    Interface function for searching the database.
    """
    if not os.path.exists('datasets/vector_index.faiss'):
        return "No database found. Please add videos first.", None
    results, top_videos = query_vector_database(query_text, embedding_model, top_k=top_k)

    # Prepare detailed results
    detailed_html = "<h1>Detailed Results:</h1>"
    for _, row in results.iterrows():
        detailed_html += f"""
        <div style='margin-bottom:20px;'>
            <img src='file/{row['thumbnail_path']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Text:</strong> {row['text']}</p>
            <p><strong>Score:</strong> {row['score']:.4f}</p>
            <p><a href='{row['YouTube_link']}' target='_blank'>Watch Video</a></p>
            <div style='clear:both;'></div>
        </div>
        """

    # Prepare top videos
    top_videos_html = "<h1>Top Relevant Videos:</h1>"
    for idx, row in top_videos.iterrows():
        top_videos_html += f"""
        <div style='margin-bottom:20px;'>
            <h4>Rank {idx +1}</h4>
            <img src='file/{row['thumbnail']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Relevance Score:</strong> {row['relevance']:.4f}</p>
            <p><strong>Example Text:</strong> {row['text']}</p>
            <p><a href='{row['original_link']}' target='_blank'>Watch Video</a></p>
            <div style='clear:both;'></div>
        </div>
        """
    #return detailed_html, top_videos_html
    return top_videos_html, detailed_html


def main():
    #parser = argparse.ArgumentParser(description="YouTube Video Search Application")
    parser = argparse.ArgumentParser(
        description="YouTube Video Search Application",
        epilog="""
Examples:
  # Add videos from a playlist
  python app.py add --type playlist --input "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

  # Add specific videos
  python app.py add --type videos --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ,https://www.youtube.com/watch?v=9bZkp7q19f0"

  # Search the database with a query
  python app.py search --query "machine learning tutorials" --top_k 5

  # Run the Gradio web interface
  python app.py ui
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command')

    # Add videos command
    parser_add = subparsers.add_parser('add', help='Add videos to the database')
    parser_add.add_argument('--type', choices=['playlist', 'videos'], required=True, help='Type of input')
    parser_add.add_argument('--input', required=True, help='Playlist URL or comma-separated video URLs')

    # Search command
    parser_search = subparsers.add_parser('search', help='Search the video database')
    parser_search.add_argument('--query', required=True, help='Search query')
    parser_search.add_argument('--top_k', type=int, default=5, help='Number of results to return')

    # Run Gradio interface
    parser_ui = subparsers.add_parser('ui', help='Run the Gradio web interface')

    args = parser.parse_args()

    if args.command == 'add':
        status = add_videos_interface(args.type, args.input)
        print(status)


    elif args.command == 'search':
        detailed_results, top_videos_html = search_interface(args.query, args.top_k)
        if isinstance(detailed_results, str):
            print(detailed_results)
        else:
            # Extract data from HTML for console output
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(detailed_results, 'html.parser')
            print("Detailed Results:\n")
            for div in soup.find_all('div'):
                title = div.find('p', text=lambda t: t and 'Title:' in t).text
                text = div.find('p', text=lambda t: t and 'Text:' in t).text
                score = div.find('p', text=lambda t: t and 'Score:' in t).text
                link = div.find('a')['href']
                print(f"{title}\n{score}\n{text}\nLink: {link}\n")

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

    else:
        # Run Gradio interface if no command is provided or 'ui' command is used
        with gr.Blocks() as demo:
            gr.Markdown("# YouTube Video Search Application")

            with gr.Tab("Add Videos"):
                gr.Markdown("### Add videos to the database")
                add_option = gr.Radio(["playlist", "videos"], label="Input Type", value="playlist")
                input_text = gr.Textbox(lines=2, placeholder="Enter playlist URL or comma-separated video URLs")
                add_button = gr.Button("Add Videos")
                add_output = gr.Textbox(label="Status")
                add_button.click(add_videos_interface, inputs=[add_option, input_text], outputs=add_output)

            with gr.Tab("Search"):
                gr.Markdown("### Search the video database")
                query_text = gr.Textbox(lines=1, placeholder="Enter your search query")
                top_k = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
                search_button = gr.Button("Search")
                detailed_results = gr.HTML()
                top_video_results = gr.HTML()
                search_button.click(search_interface, inputs=[query_text, top_k], outputs=[detailed_results, top_video_results])

        demo.launch()

if __name__ == "__main__":
    main()
