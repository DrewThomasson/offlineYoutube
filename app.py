import os
import pandas as pd
import argparse
import gradio as gr
from lib.functions import (
    get_video_links, 
    process_videos, 
    save_dataset, 
    create_vector_database, 
    query_vector_database
)

def main(mode, video_links=None, query=None):
    if mode == 'add_videos':
        if not video_links:
            return "No videos provided."
        data = process_videos(video_links)
        save_dataset(data)
        create_vector_database(data)
        return "Videos added and database updated successfully."
    elif mode == 'query':
        if not query:
            return "No query provided."
        results, top_videos = query_vector_database(query)
        return results, top_videos

def gradio_interface():
    def add_videos_interface(playlist_or_links):
        video_links = get_video_links(playlist_or_links)
        return main('add_videos', video_links=video_links)

    def query_interface(query):
        results, top_videos = main('query', query=query)

        # Format the detailed results with video titles and links
        detailed_results = "\n\n".join([
            f"**Video Title:** {row['video_title']}\n"
            f"**Text:** {row['text']}\n"
            f"**Link:** [Watch Video]({row['YouTube_link']})\n"
            f"**Score:** {row['score']:.4f}"
            for _, row in results.iterrows()
        ])

        # Format the top videos with thumbnails, links, and relevance scores
        top_videos_markdown = "\n\n".join([
            f"### Rank {idx + 1}\n"
            f"**Video Title:** {row['video_title']}\n"
            f"**Relevance Score:** {row['relevance']:.4f}\n"
            f"**Link:** [Watch Video]({row['original_link']})\n"
            f"![Thumbnail]({row['thumbnail']})\n"
            f"**Example Text:** {row['text']}"
            for idx, row in top_videos.iterrows()
        ])

        return detailed_results, top_videos_markdown

    with gr.Blocks() as demo:
        gr.Markdown("## YouTube Video Dataset Manager")

        with gr.Tab("Add Videos"):
            playlist_or_links = gr.Textbox(
                label="Enter Playlist URL or Video Links (comma-separated)"
            )
            add_button = gr.Button("Add to Dataset")
            output_add = gr.Textbox(label="Status")
            add_button.click(add_videos_interface, inputs=[playlist_or_links], outputs=[output_add])

        with gr.Tab("Query Database"):
            query_input = gr.Textbox(label="Enter Query")
            query_button = gr.Button("Search")
            result_output = gr.Markdown(label="Search Results")
            top_videos_output = gr.Markdown(label="Top Videos")
            query_button.click(
                query_interface, 
                inputs=[query_input], 
                outputs=[result_output, top_videos_output]
            )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Dataset Manager")
    parser.add_argument("--mode", choices=["cli", "gradio"], help="Run mode: cli or gradio")
    parser.add_argument("--action", choices=["add_videos", "query"], help="Action to perform in CLI mode")
    parser.add_argument("--links", help="Playlist URL or comma-separated video links")
    parser.add_argument("--query", help="Query to search in the existing database")

    args = parser.parse_args()

    if args.mode == "gradio":
        gradio_interface()
    elif args.mode == "cli":
        if args.action == "add_videos":
            if not args.links:
                print("Error: Please provide a playlist URL or video links with --links")
            else:
                video_links = get_video_links(args.links)
                print(main('add_videos', video_links=video_links))

        elif args.action == "query":
            if not args.query:
                print("Error: Please provide a query with --query")
            else:
                results, top_videos = main('query', query=args.query)

                # Print detailed search results (relevant sentences)
                print("\nDetailed Results:\n")
                for _, row in results.iterrows():
                    print(f"Video Title: {row['video_title']}")
                    print(f"Text: {row['text']}")
                    print(f"Link: {row['YouTube_link']}")
                    print(f"Thumbnail: {row['thumbnail_path']}")
                    print(f"Score: {row['score']:.4f}\n")

                # Print top-ranked videos based on relevance
                print("\nTop Relevant Videos:\n")
                for idx, row in top_videos.iterrows():
                    print(f"Rank {idx + 1}:")
                    print(f"Video Title: {row['video_title']}")
                    print(f"Relevance Score: {row['relevance']:.4f}")
                    print(f"Video Link: {row['original_link']}")
                    print(f"Thumbnail: {row['thumbnail']}")
                    print(f"Example Text: {row['text']}\n")

        else:
            print("Invalid action. Use --action add_videos or --action query.")


    else:
        print("Please provide a valid mode. Use --help for more details.")
