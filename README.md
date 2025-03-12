# Audio/Video Transcription App

This Streamlit application allows users to upload audio or video files and get their transcription using Google's Speech-to-Text API.

## Features

- Support for multiple audio and video formats (WAV, MP3, MP4, AVI, MOV, OGG, WEBM)
- Automatic conversion of video files to audio
- Automatic conversion of audio files to WAV format
- Real-time transcription using Google Cloud Speech-to-Text
- Clean and simple user interface

## Prerequisites

1. Python 3.7 or higher
2. Google Cloud account with Speech-to-Text API enabled
3. Google Cloud credentials (service account key)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials:
   - Create a project in Google Cloud Console
   - Enable the Speech-to-Text API
   - Create a service account and download the JSON key file
   - Rename the key file to `google_credentials.json` and place it in the project root
   - Create a `.env` file and add:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json
     ```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload an audio or video file using the file uploader

4. Wait for the transcription to complete

## Supported File Formats

- Audio: WAV, MP3, OGG
- Video: MP4, AVI, MOV, WEBM

## Notes

- For large files, the transcription process might take some time
- Make sure your audio is clear and has minimal background noise for better results
- The application currently supports English language only (can be modified in the code)
- Temporary files are automatically cleaned up after processing 