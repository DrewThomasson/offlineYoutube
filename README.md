# vectorDatabaseYoutube
Easy searchable Vector database of YouTube playlist

## Overview  
This application vectorizes YouTube videos for similarity searches using embeddings. It allows quick retrieval of related videos based on input queries.

---

## Installation

- Tested on `Python 3.10.15`

1. **Clone the repository**  
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies directly**  
   Run the following in your terminal:  
   ```bash
   pip install yt-dlp pandas numpy requests faiss-cpu faster-whisper sentence-transformers
   ```

---

## Usage

1. **Run the script**  
   ```bash
   python VectorDatabaseYoutube.py
   ```

2. **Expected Inputs**:  
   - A collection of YouTube video URLs.
   - Queries for retrieving similar content.

3. **Example Output**:  
   ```
   Input Query: "Videos similar to Deep Learning tutorials"
   
   Output:
   1. "Deep Learning Crash Course" - Score: 0.92  
   2. "Neural Networks Basics" - Score: 0.89  
   3. "Introduction to AI & Deep Learning" - Score: 0.85  
   ```

---

## Features  
- **Content-based video search** with embeddings.  
- **Fast similarity search** across large video collections.  
- **Scalable** for extensive YouTube datasets.

---

## Contribution  
Fork, improve, and submit pull requests to help enhance this tool!

---

## License  
Licensed under the MIT License.
