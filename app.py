import os
import tempfile
from pathlib import Path
import ffmpeg
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import wave
import time
from datetime import datetime
from langdetect import detect
import queue
import threading
import io
import google.auth
import google.auth.transport.requests
import requests
from googleapiclient.discovery import build
from google.cloud import speech
try:
    # Try the newer way first
    from google.cloud import translate_v2 as translate
except ImportError:
    # Fall back to the direct import if the v2 module isn't available
    from google.cloud import translate
import yt_dlp

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-account.json'

# Supported languages
LANGUAGES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali',
    'ca': 'Catalan', 'cs': 'Czech', 'da': 'Danish', 'de': 'German',
    'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian',
    'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 'gu': 'Gujarati',
    'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian',
    'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese',
    'kn': 'Kannada', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'nl': 'Dutch',
    'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 'sr': 'Serbian',
    'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian',
    'ur': 'Urdu', 'vi': 'Vietnamese', 'zh': 'Chinese'
}

# Urdu language for priority selection
URDU_LANGUAGE = "ur - Urdu"

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.overflow_count = 0
        self.max_overflow = 5

    def callback(self, indata, frames, time, status):
        if status:
            if status.input_overflow:
                self.overflow_count += 1
                if self.overflow_count > self.max_overflow:
                    self.is_recording = False
                    return
            else:
                print(f"Status: {status}")
        
        if self.is_recording:  # Only add data if still recording
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.overflow_count = 0
        
        def record():
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.callback,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                    dtype=np.float32
                ):
                    while self.is_recording:
                        if not self.audio_queue.empty():
                            data = self.audio_queue.get()
                            # Convert float32 to int16
                            data = (data * 32767).astype(np.int16)
                            self.audio_data.append(data)
                        time.sleep(0.001)  # Small sleep to prevent CPU overuse
            except Exception as e:
                print(f"Recording error: {e}")
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        if not self.audio_data:
            return np.array([], dtype=np.int16)
        
        return np.concatenate(self.audio_data)

def save_audio_to_wav(audio_data, sample_rate):
    """Save recorded audio to WAV file."""
    if len(audio_data) == 0:
        print("No audio data recorded!")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return filename

def convert_video_to_audio(video_path):
    """Extract audio from video file using ffmpeg."""
    audio_path = str(video_path).rsplit(".", 1)[0] + ".wav"
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise

def convert_audio_to_wav(audio_path):
    """Convert audio file to WAV format."""
    audio = AudioSegment.from_file(audio_path)
    wav_path = str(audio_path).rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return wav_path

def translate_text(text, target_language="en"):
    """Translate text using Google Cloud Translation API for better accuracy."""
    try:
        # Initialize the Translation client
        translate_client = translate.Client()
        
        # Perform the translation
        try:
            # For translate_v2
            result = translate_client.translate(text, target_language=target_language)
            return result["translatedText"]
        except (TypeError, KeyError):
            # For newer translate versions
            result = translate_client.translate(text, target_language=target_language)
            if isinstance(result, dict) and "translatedText" in result:
                return result["translatedText"]
            elif hasattr(result, "translated_text"):
                return result.translated_text
            else:
                return str(result)
        
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return text  # Return original text if translation fails

def transcribe_audio(audio_path, language_code="en-US", translate_to_english=True, progress_callback=None):
    """
    Transcribe audio and optionally translate to English.
    
    Args:
        audio_path: Path to the audio file
        language_code: Language code for transcription
        translate_to_english: Whether to translate to English
        progress_callback: Function to call with progress updates
          This should accept (message, progress_value) parameters
    """
    client = speech.SpeechClient()
    
    try:
        # Load the audio file
        audio_segment = AudioSegment.from_wav(audio_path)
        
        # Calculate total duration in seconds
        duration_seconds = len(audio_segment) / 1000
        
        # Split audio into 45-second chunks (safely under Google's 60s limit)
        chunk_length_ms = 45 * 1000  # 45 seconds
        chunks = [audio_segment[i:i + chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
        
        if progress_callback:
            progress_callback(f"Audio split into {len(chunks)} chunks for processing.", 0)
        
        # Process each chunk and combine transcriptions
        transcript = ""
        
        for i, chunk in enumerate(chunks):
            # Export chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_path = temp_file.name
                chunk.export(chunk_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            
            try:
                # Load the temporary file for transcription
                with open(chunk_path, "rb") as audio_file:
                    content = audio_file.read()
                
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                )
                
                # Process this chunk
                response = client.recognize(config=config, audio=audio)
                
                # Add results to transcript
                for result in response.results:
                    transcript += result.alternatives[0].transcript + " "
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(chunks)
                    progress_callback(f"Processing chunk {i+1} of {len(chunks)} ({int(progress*100)}% complete)", progress)
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(chunk_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {chunk_path}: {e}")
        
        transcript = transcript.strip()
        
        # Check if we need to translate
        if translate_to_english and not language_code.startswith("en"):
            if progress_callback:
                progress_callback("Transcription complete. Translating to English...", 0.9)
            translated_transcript = translate_text(transcript)
            return {
                "original_transcript": transcript,
                "translated_transcript": translated_transcript
            }
        else:
            return {
                "original_transcript": transcript, 
                "translated_transcript": transcript
            }
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise e

def detect_language(text):
    """Detect language of the text and return appropriate language code."""
    try:
        detected = detect(text)
        return f"{detected}-{detected.upper()}" if detected in LANGUAGES else "en-US"
    except:
        return "en-US"

def download_audio_from_youtube(youtube_url):
    """Download audio from a YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': tempfile.gettempdir() + '/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        audio_file_path = ydl.prepare_filename(info_dict)
        return audio_file_path.replace('.webm', '.wav')  # Adjust extension if needed 
