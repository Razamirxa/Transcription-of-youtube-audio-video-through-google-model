import os
import tempfile
from pathlib import Path
import streamlit as st
import time
from datetime import datetime

# Import core functionality from app.py
from app import (
    AudioRecorder, 
    save_audio_to_wav, 
    convert_video_to_audio, 
    convert_audio_to_wav, 
    translate_text, 
    transcribe_audio, 
    detect_language, 
    download_audio_from_youtube, 
    LANGUAGES, 
    URDU_LANGUAGE
)

# Set page configuration
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .recording-indicator {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2rem;
        animation: blink 1.5s linear infinite;
    }
    @keyframes blink {
        50% {
            opacity: 0.5;
        }
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .transcript-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0D47A1;
        margin-top: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .translation-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #43A047;
        margin-top: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def update_progress(message, progress_value):
    """Update progress UI elements with progress information"""
    progress_placeholder = st.session_state.get('progress_placeholder')
    progress_bar = st.session_state.get('progress_bar')
    
    if progress_placeholder:
        progress_placeholder.text(message)
    
    if progress_bar and progress_value is not None:
        progress_bar.progress(progress_value)
    
    # If we're done, clear the progress indicators
    if progress_value == 1.0:
        if progress_placeholder:
            progress_placeholder.empty()
        if progress_bar:
            progress_bar.empty()

def handle_microphone_recording():
    """Handle microphone recording functionality"""
    st.markdown('<div class="sub-header">Microphone Recording</div>', unsafe_allow_html=True)
    
    # Initialize session state for recorder
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", key="start_rec", disabled=st.session_state.recording):
            st.session_state.recorder.start_recording()
            st.session_state.recording = True
            st.session_state.audio_data = None
    
    with col2:
        if st.button("⏹️ Stop Recording", key="stop_rec", disabled=not st.session_state.recording):
            audio_data = st.session_state.recorder.stop_recording()
            st.session_state.recording = False
            st.session_state.audio_data = audio_data
    
    # Recording indicator
    if st.session_state.recording:
        st.markdown('<div class="recording-indicator">🔴 Recording in progress...</div>', unsafe_allow_html=True)
    
    # Process the recorded audio
    if st.session_state.audio_data is not None:
        audio_path = None
        try:
            # Save to WAV file
            audio_path = save_audio_to_wav(st.session_state.audio_data, st.session_state.recorder.sample_rate)
            
            if audio_path:
                # Display audio player
                st.audio(audio_path)
                
                # Get language code
                language_code = st.session_state.selected_language.split(" - ")[0]
                
                # Setup progress tracking
                st.session_state.progress_placeholder = st.empty()
                st.session_state.progress_bar = st.progress(0)
                
                # Transcribe the audio
                with st.spinner("Transcribing and translating..."):
                    transcript_result = transcribe_audio(
                        audio_path, 
                        language_code, 
                        st.session_state.translate_to_english,
                        progress_callback=update_progress
                    )
                    
                    # Display the original transcript
                    st.markdown(f'<div class="sub-header">Original Transcript ({st.session_state.selected_language.split(" - ")[1]})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="transcript-box">{transcript_result["original_transcript"]}</div>', unsafe_allow_html=True)
                    
                    # If translation is different from original, display it
                    if st.session_state.translate_to_english and transcript_result["original_transcript"] != transcript_result["translated_transcript"]:
                        st.markdown('<div class="sub-header">English Translation</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="translation-box">{transcript_result["translated_transcript"]}</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up the audio file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {audio_path}: {e}")
            st.session_state.audio_data = None

def handle_file_upload():
    """Handle file upload functionality"""
    st.markdown('<div class="sub-header">Upload Audio/Video File</div>', unsafe_allow_html=True)
    
    # File uploader for audio/video files
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["wav", "mp3", "mp4", "m4a", "flac", "ogg", "webm"])
    
    if uploaded_file is not None:
        file_path = Path(tempfile.gettempdir()) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.markdown('<div class="info-box">File uploaded successfully! Processing...</div>', unsafe_allow_html=True)
        
        # Convert to WAV if necessary
        if file_path.suffix.lower() in [".mp4", ".m4a", ".webm"]:
            audio_path = convert_video_to_audio(file_path)
        else:
            audio_path = convert_audio_to_wav(file_path)
        
        # Display audio player
        st.audio(audio_path)
        
        # Get language code
        language_code = st.session_state.selected_language.split(" - ")[0]
        
        # Setup progress tracking
        st.session_state.progress_placeholder = st.empty()
        st.session_state.progress_bar = st.progress(0)
        
        # Transcribe the audio
        with st.spinner("Transcribing and translating..."):
            transcript_result = transcribe_audio(
                audio_path, 
                language_code, 
                st.session_state.translate_to_english,
                progress_callback=update_progress
            )
            
            # Display the original transcript
            st.markdown(f'<div class="sub-header">Original Transcript ({st.session_state.selected_language.split(" - ")[1]})</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-box">{transcript_result["original_transcript"]}</div>', unsafe_allow_html=True)
            
            # If translation is different from original, display it
            if st.session_state.translate_to_english and transcript_result["original_transcript"] != transcript_result["translated_transcript"]:
                st.markdown('<div class="sub-header">English Translation</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="translation-box">{transcript_result["translated_transcript"]}</div>', unsafe_allow_html=True)

def handle_youtube_url():
    """Handle YouTube URL functionality"""
    st.markdown('<div class="sub-header">YouTube Video Transcription</div>', unsafe_allow_html=True)
    
    # YouTube URL input
    youtube_url = st.text_input("🎬 Enter YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if youtube_url:
        try:
            st.markdown('<div class="info-box">Processing YouTube video... This may take a while for longer videos.</div>', unsafe_allow_html=True)
            
            # Download the audio from YouTube
            audio_file_path = download_audio_from_youtube(youtube_url)
            
            # Convert to WAV if necessary
            audio_path = convert_audio_to_wav(audio_file_path)
            
            # Display audio player
            st.audio(audio_path)
            
            # Get language code
            language_code = st.session_state.selected_language.split(" - ")[0]
            
            # Setup progress tracking
            st.session_state.progress_placeholder = st.empty()
            st.session_state.progress_bar = st.progress(0)
            
            # Transcribe the audio
            with st.spinner("Transcribing and translating..."):
                transcript_result = transcribe_audio(
                    audio_path, 
                    language_code, 
                    st.session_state.translate_to_english,
                    progress_callback=update_progress
                )
                
                # Display the original transcript
                st.markdown(f'<div class="sub-header">Original Transcript ({st.session_state.selected_language.split(" - ")[1]})</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="transcript-box">{transcript_result["original_transcript"]}</div>', unsafe_allow_html=True)
                
                # If translation is different from original, display it
                if st.session_state.translate_to_english and transcript_result["original_transcript"] != transcript_result["translated_transcript"]:
                    st.markdown('<div class="sub-header">English Translation</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="translation-box">{transcript_result["translated_transcript"]}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the YouTube video: {str(e)}")

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/microphone.png", width=80)
    st.sidebar.title("Audio Transcription")
    
    # Language selection
    record_language_options = [URDU_LANGUAGE] + [f"{code} - {name}" for code, name in LANGUAGES.items() if code != "ur"]
    selected_language = st.sidebar.selectbox(
        "Select language",
        record_language_options,
        key="sidebar_language"
    )
    
    # Store selected language in session state
    if 'selected_language' not in st.session_state or st.session_state.selected_language != selected_language:
        st.session_state.selected_language = selected_language
    
    # Translation option
    translate_to_english = st.sidebar.checkbox("Translate to English", value=True, key="sidebar_translate")
    
    # Store translation option in session state
    if 'translate_to_english' not in st.session_state or st.session_state.translate_to_english != translate_to_english:
        st.session_state.translate_to_english = translate_to_english
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["Microphone Recording", "File Upload", "YouTube URL"]
    )
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application uses Google Cloud Speech-to-Text API to transcribe audio from various sources. "
        "It supports multiple languages and can translate the transcription to English."
    )
    
    # Help section in sidebar
    st.sidebar.markdown("### Help")
    with st.sidebar.expander("How to use"):
        st.markdown("""
        1. Select the language of the audio from the dropdown.
        2. Choose whether to translate to English.
        3. Select an input method:
           - **Microphone Recording**: Record directly from your microphone.
           - **File Upload**: Upload an audio or video file.
           - **YouTube URL**: Enter a YouTube video URL to transcribe.
        4. Wait for the transcription and translation (if selected).
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">Audio Transcription & Translation App</h1>', unsafe_allow_html=True)
    
    # Display different content based on selected input method
    if input_method == "Microphone Recording":
        handle_microphone_recording()
    elif input_method == "File Upload":
        handle_file_upload()
    elif input_method == "YouTube URL":
        handle_youtube_url()

if __name__ == "__main__":
    main() 
