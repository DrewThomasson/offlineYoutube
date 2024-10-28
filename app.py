# app.py

import gradio as gr
from lib.functions import initialize_models, setup_directories, process_videos, save_dataset, create_vector_database, query_vector_database, get_video_links
import os

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
    create_vector_database(data, embedding_model)
    return "Videos processed and database updated."

def search_interface(query_text, top_k):
    """
    Interface function for searching the database.
    """
    if not os.path.exists('datasets/vector_index.faiss'):
        return "No database found. Please add videos first.", None
    results, top_videos = query_vector_database(query_text, embedding_model, top_k=top_k)

    # Prepare detailed results
    detailed_html = "<h3>Detailed Results:</h3>"
    for _, row in results.iterrows():
        detailed_html += f"""
        <div style='margin-bottom:20px;'>
            <img src='{row['thumbnail_path']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Text:</strong> {row['text']}</p>
            <p><strong>Score:</strong> {row['score']:.4f}</p>
            <p><a href='{row['YouTube_link']}' target='_blank'>Watch Video</a></p>
            <div style='clear:both;'></div>
        </div>
        """

    # Prepare top videos
    top_videos_html = "<h3>Top Relevant Videos:</h3>"
    for idx, row in top_videos.iterrows():
        top_videos_html += f"""
        <div style='margin-bottom:20px;'>
            <h4>Rank {idx +1}</h4>
            <img src='{row['thumbnail']}' alt='Thumbnail' width='120' style='float:left; margin-right:10px;'>
            <p><strong>Title:</strong> {row['video_title']}</p>
            <p><strong>Relevance Score:</strong> {row['relevance']:.4f}</p>
            <p><strong>Example Text:</strong> {row['text']}</p>
            <p><a href='{row['original_link']}' target='_blank'>Watch Video</a></p>
            <div style='clear:both;'></div>
        </div>
        """
    return detailed_html, top_videos_html

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

if __name__ == "__main__":
    demo.launch()
