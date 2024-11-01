# **Offline YouTube Video Search Application**

This application allows users to **extract transcripts from YouTube videos**, **upload their own video/audio files**, **create searchable vector databases**, and **perform semantic searches** using a **Gradio web interface** or **command-line interface (CLI)**. It's powered by `faster-whisper` for transcription, `FAISS` for vector search, and `sentence-transformers` for text embeddings.

---

## **Features**

- Extract transcripts from individual videos, playlists, and entire channels.
- **Upload your own video or audio files for processing.**
- Automatically detect playlists, channels, and individual video links.
- Automatically download video thumbnails.
- Store transcripts and create a searchable vector database.
- Perform semantic searches on video content.
- Supports **Gradio web interface** and **CLI** for flexible usage.
- Easily add more videos or your own files to the dataset.

---

## **Web Interface**

### **Add Videos Tab**

- **Enter playlist, channel, and/or video URLs (comma-separated).**
- **Upload your own video/audio files.**
- **Option to process entire channels when a channel URL is provided.**
- **Option to keep videos stored locally or not.**

<img width="520" alt="Add Videos Tab" src="path/to/your/add_videos_tab_screenshot.png">

### **Search Tab**

- **Enter your search query to find relevant snippets.**
- **View top relevant videos with thumbnails and play local videos if available.**
- **View detailed results with timestamps and direct links.**

<img width="526" alt="Search Tab" src="path/to/your/search_tab_screenshot.png">

---

## **Installation**

Ensure you have Python installed (>= 3.7). Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## **Usage**

The app provides **two ways to interact**:  
1. **Gradio Web Interface**  
2. **Command-Line Interface (CLI)**

### **1. Running the Gradio Web Interface**

Launch the web interface:

```bash
python app.py ui
```

or simply:

```bash
python app.py
```

Then, open the URL (usually `http://127.0.0.1:7860`) in your browser.

#### **Gradio Interface Tabs:**

- **Add Videos:**  
  - Enter playlist URLs, channel URLs, and/or individual video URLs (comma-separated).
  - **Upload your own video or audio files for processing.**
  - **Option to process entire YouTube channels when a channel URL is provided.**
  - **Option to keep videos stored locally or not.**
  - The app will automatically detect whether each link is a playlist, channel, or a video.
  - Videos and uploaded files will be transcribed, and the database will be updated with the content.
  
- **Search:**  
  - Enter search queries to find relevant snippets from the video transcripts.
  - Results are ranked based on semantic similarity and include video thumbnails.
  - **If local videos are available, you can play them directly in the interface.**

---

### **2. Command-Line Interface (CLI)**

The CLI provides more flexibility for programmatic use.

#### **Commands Overview**

Use the `--help` command to view available commands and examples:

```bash
python app.py --help
```

**Output:**

```
usage: app.py [-h] {add,search,ui} ...

YouTube Video Search Application

positional arguments:
  {add,search,ui}   Available commands
    add             Add videos to the database
    search          Search the video database
    ui              Run the Gradio web interface

optional arguments:
  -h, --help        Show this help message and exit

Examples:
  # Add videos from a playlist and keep videos locally
  python app.py add --input "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" --keep_videos

  # Add specific videos without keeping videos locally
  python app.py add --input "https://www.youtube.com/watch?v=VIDEO_ID1,https://www.youtube.com/watch?v=VIDEO_ID2"

  # Add videos from a channel (process entire channel)
  python app.py add --input "https://www.youtube.com/channel/CHANNEL_ID" --process_channel

  # Search the database with a query
  python app.py search --query "Your search query" --top_k 5

  # Run the Gradio web interface
  python app.py ui
```

---

### **Examples of CLI Usage**

#### **1. Adding Videos**

- **Add Playlists and Videos:**

   ```bash
   python app.py add --input "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID,https://www.youtube.com/watch?v=VIDEO_ID"
   ```

- **Add Specific Videos Without Keeping Them Locally:**

   ```bash
   python app.py add --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ,https://www.youtube.com/watch?v=9bZkp7q19f0"
   ```

- **Add Videos from a Channel (Process Entire Channel):**

   ```bash
   python app.py add --input "https://www.youtube.com/channel/CHANNEL_ID" --process_channel
   ```

- **Add Videos and Keep Videos Stored Locally:**

   ```bash
   python app.py add --input "https://www.youtube.com/watch?v=VIDEO_ID" --keep_videos
   ```

#### **2. Searching the Database**

- **Perform a Search:**

   ```bash
   python app.py search --query "machine learning tutorials" --top_k 5
   ```

---

### **How It Works**

1. **Adding Videos and Uploaded Files:**
   - The app accepts a list of links and automatically detects whether each link is a playlist, channel, or an individual video.
   - **You can upload your own video or audio files for processing.**
   - It downloads video audio (or uses uploaded files) and transcribes it using `faster-whisper`.
   - Thumbnails are downloaded and saved locally.
   - The transcript data is saved in `datasets/transcript_dataset.csv`.
   - A vector database is updated using FAISS with embeddings generated by `sentence-transformers`.

2. **Incremental Updating:**
   - Videos and uploaded files are processed one by one, and the dataset and vector database are updated incrementally.
   - This ensures efficient processing, especially when dealing with large datasets.

3. **Searching the Database:**
   - When a query is entered, the app computes its embedding and searches the FAISS index for relevant video snippets.
   - The top results are displayed with thumbnails, titles, and links to the videos.
   - **If local videos are available, you can play them directly in the interface.**

---

### **FAQ**

#### **1. How do I add multiple playlists, channels, and videos at once?**

Simply provide a comma-separated list of URLs, and the app will automatically detect and process each link:

```bash
python app.py add --input "https://www.youtube.com/playlist?list=PLAYLIST_ID1,https://www.youtube.com/watch?v=VIDEO_ID,https://www.youtube.com/channel/CHANNEL_ID"
```

If you want to process entire channels, make sure to include the `--process_channel` flag:

```bash
python app.py add --input "https://www.youtube.com/channel/CHANNEL_ID" --process_channel
```

#### **2. How can I upload my own video or audio files for processing?**

In the Gradio web interface, navigate to the **Add Videos** tab. Use the **"Upload your own video/audio files"** option to upload one or multiple files. The app will process these files and add them to the database.

#### **3. Why aren’t new videos or uploaded files showing up in search results?**

Ensure that the videos or files have been fully processed and that the vector database has been updated. The app handles this automatically, but processing may take time for large videos, playlists, or channels.

#### **4. How do I prevent videos from being stored locally?**

By default, the app keeps videos stored locally. To change this behavior, use the `--keep_videos` flag and set it to `False`:

```bash
python app.py add --input "VIDEO_OR_PLAYLIST_URL" --keep_videos False
```

In the Gradio interface, uncheck the **"Keep videos stored locally"** option in the **Add Videos** tab.

#### **5. Can I process entire YouTube channels?**

Yes! Use the `--process_channel` flag when adding videos via the CLI:

```bash
python app.py add --input "https://www.youtube.com/channel/CHANNEL_ID" --process_channel
```

In the Gradio interface, check the **"Process entire channel when a channel URL is provided"** option in the **Add Videos** tab.

#### **6. Can I search the database without launching the Gradio interface?**

Yes! Use the `search` command via the CLI:

```bash
python app.py search --query "Your query" --top_k 5
```

---

### **Project Structure**

```
.
├── app.py                       # Main application script (Gradio + CLI)
├── functions.py                 # Helper functions for transcription, FAISS, etc.
├── datasets/
│   ├── transcript_dataset.csv   # CSV file storing transcripts
│   └── vector_index.faiss       # FAISS vector index
├── thumbnails/                  # Folder for storing video thumbnails
├── videos/                      # Folder for storing downloaded videos (if keep_videos is True)
├── tmp/                         # Temporary folder for videos (if keep_videos is False)
├── uploaded_files/              # Folder for storing uploaded files
```

---

### **Known Limitations**

- **Processing Time:** Transcribing videos and generating embeddings can be time-consuming, especially for long videos, large playlists, or channels.
- **Storage Requirements:** Keeping videos stored locally will require additional disk space. Use the `--keep_videos False` option if storage is a concern.
- **Large Datasets:** As the dataset grows, querying may take longer. Consider optimizing the FAISS index for very large datasets.

---

### **Contributing**

Feel free to fork the repository, open issues, or submit pull requests if you'd like to contribute to this project.

---

### **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---

### **Acknowledgments**

- **faster-whisper** for fast transcription.
- **FAISS** for efficient vector search.
- **Gradio** for the interactive web interface.
- **yt-dlp** for downloading video content.

---
