#!/usr/bin/env python3
"""
ElevenLabs Transcription Service
-------------------------------
Integration with ElevenLabs API for high-quality transcription.

This module handles the transcription of audio files using ElevenLabs' API,
providing accurate transcription with word-level timestamps.
"""
import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed text with timing information"""
    start_time: float
    end_time: float
    text: str
    words: List[Dict[str, Any]] = None


class ElevenLabsTranscriptionService:
    """Transcription service using ElevenLabs API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "eleven-turbo-2"):
        """
        Initialize ElevenLabs transcription service
        
        Args:
            api_key: ElevenLabs API key
            model: Transcription model to use
        """
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.model = model
        self.headers = {
            "xi-api-key": self.api_key,
            "Accept": "application/json"
        }
        
        if not self.api_key:
            logger.warning("No ElevenLabs API key provided, transcription will fail")
        else:
            logger.info("ElevenLabs transcription service initialized successfully")
    
    def transcribe(self, audio_path: Union[str, Path], 
                  languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using ElevenLabs API
        
        Args:
            audio_path: Path to audio file
            languages: Optional list of language codes
            
        Returns:
            Transcription result with segments and metadata
        """
        start_time = time.time()
        
        # Ensure audio path is a Path object
        audio_path = Path(audio_path)
        
        # Prepare API endpoint
        url = f"{ELEVENLABS_API_URL}/speech-recognition/transcription"
        
        # Set up parameters
        params = {
            "model_id": self.model,
            "transcription_format": "json"
        }
        
        # Add language detection if specified
        if languages:
            params["detect_language"] = "true"
            languages_str = ",".join(languages)
            params["language"] = languages_str
        
        # Prepare files
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/mpeg")}
            
            # Make API request
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    params=params,
                    files=files
                )
                
                if response.status_code == 200:
                    # Parse the JSON response
                    transcription = response.json()
                    
                    # Format the result into our standard structure
                    formatted_result = self._format_transcription(transcription)
                    
                    # Log processing stats
                    audio_duration = formatted_result.get("duration", 0)
                    processing_time = time.time() - start_time
                    processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0
                    
                    logger.info(f"Transcription completed in {processing_time:.2f}s for {audio_duration:.2f}s audio "
                               f"(processing ratio: {processing_ratio:.2f}x)")
                    
                    return formatted_result
                else:
                    logger.error(f"Transcription failed: {response.status_code} - {response.text}")
                    return self._create_empty_result()
                    
            except Exception as e:
                logger.error(f"Exception during transcription: {str(e)}")
                return self._create_empty_result()
    
    def _format_transcription(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format ElevenLabs API response into standardized format
        
        Args:
            result: API response from ElevenLabs
            
        Returns:
            Formatted transcription result
        """
        # Initialize segments list and metadata
        segments = []
        words = []
        
        # Extract transcript text
        text = result.get("text", "")
        
        # Extract itn_text if available (inverse text normalization)
        itn_text = result.get("itn_text", text)
        
        # Extract metadata
        language = result.get("language", "en")
        
        # Process words with timestamps
        word_entries = result.get("words", [])
        
        # Group words into sentences/segments
        # ElevenLabs doesn't provide segment information directly,
        # so we need to create segments based on punctuation and pauses
        current_segment = {
            "text": "",
            "words": [],
            "start_time": None,
            "end_time": None
        }
        
        segment_break_chars = ['.', '!', '?', '\n']
        max_pause_threshold = 0.8  # Pause between words in seconds to create a new segment
        
        last_end_time = 0
        
        for word_entry in word_entries:
            word = word_entry.get("text", "")
            start = word_entry.get("start_time", 0)
            end = word_entry.get("end_time", 0)
            
            # Store full word data
            words.append({
                "word": word,
                "start_time": start,
                "end_time": end
            })
            
            # Initialize segment start time if not set
            if current_segment["start_time"] is None:
                current_segment["start_time"] = start
            
            # Check if we should break to a new segment
            create_new_segment = False
            
            # Check for long pause between words
            if words and (start - last_end_time) > max_pause_threshold:
                create_new_segment = True
            
            # Check for punctuation that indicates end of sentence
            if any(char in word for char in segment_break_chars):
                create_new_segment = True
            
            # Add word to current segment
            current_segment["text"] += " " + word if current_segment["text"] else word
            current_segment["words"].append({
                "word": word,
                "start_time": start,
                "end_time": end
            })
            current_segment["end_time"] = end
            
            # Remember the end time of this word
            last_end_time = end
            
            # Create new segment if needed
            if create_new_segment and current_segment["text"]:
                segments.append(TranscriptionSegment(
                    start_time=current_segment["start_time"],
                    end_time=current_segment["end_time"],
                    text=current_segment["text"].strip(),
                    words=current_segment["words"]
                ))
                
                # Reset for next segment
                current_segment = {
                    "text": "",
                    "words": [],
                    "start_time": None,
                    "end_time": None
                }
        
        # Add the final segment if not empty
        if current_segment["text"]:
            segments.append(TranscriptionSegment(
                start_time=current_segment["start_time"],
                end_time=current_segment["end_time"],
                text=current_segment["text"].strip(),
                words=current_segment["words"]
            ))
        
        # Calculate audio duration from the last word's end time
        duration = max(word["end_time"] for word in words) if words else 0
        
        # Create the formatted result
        formatted_result = {
            "text": text,
            "itn_text": itn_text,
            "language": language,
            "duration": duration,
            "segments": segments,
            "words": words,
            "num_segments": len(segments),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return formatted_result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result structure for failed transcriptions
        
        Returns:
            Empty transcription result
        """
        return {
            "text": "",
            "itn_text": "",
            "language": "en",
            "duration": 0,
            "segments": [],
            "words": [],
            "num_segments": 0,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": "Transcription failed"
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python elevenlabs.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    transcriber = ElevenLabsTranscriptionService()
    
    result = transcriber.transcribe(audio_path)
    
    print(f"Transcribed text: {result['text'][:100]}...")
    print(f"Number of segments: {result['num_segments']}")
    print(f"Duration: {result['duration']:.2f}s")