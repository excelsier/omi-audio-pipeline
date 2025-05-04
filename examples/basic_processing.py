#!/usr/bin/env python3
"""
Basic OMI Audio Pipeline Example
-------------------------------
Demonstrates how to use the OMI Audio Pipeline to process an audio file.
"""
import os
import logging
from pathlib import Path

from omi_audio.pipeline.complete_pipeline import CompletePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Process an audio file with the OMI Audio Pipeline"""
    # Check for required API keys
    if not os.environ.get("PYANNOTE_API_KEY"):
        logger.error("PYANNOTE_API_KEY environment variable is not set")
        logger.error("Please set it to use the diarization functionality")
        return
    
    if not os.environ.get("ELEVENLABS_API_KEY"):
        logger.warning("ELEVENLABS_API_KEY environment variable is not set")
        logger.warning("Using default transcription service instead of ElevenLabs")
    
    # Path to your audio file
    audio_path = "path/to/your/audio/file.mp3"
    
    # Check if the audio file exists
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.error("Please update the audio_path variable to a valid audio file path")
        return
    
    # Initialize the pipeline with customized parameters
    pipeline = CompletePipeline(
        min_speakers=2,         # Minimum number of speakers to detect
        max_speakers=8,         # Maximum number of speakers to detect
        clustering_threshold=0.7 # Threshold for speaker clustering
    )
    
    # Process the audio file
    result = pipeline.process_audio(
        audio_path=audio_path,
        languages=["en"],                 # Language for transcription
        generate_visualization=True,      # Generate HTML visualization
        update_speaker_profiles=True      # Extract and save speaker profiles
    )
    
    # Print results
    print(f"Processing complete!")
    print(f"Found {result.diarization.get('num_speakers', 0)} speakers in the audio")
    
    # Output file locations
    if result.json_path:
        print(f"Results saved to: {result.json_path}")
    if result.html_path:
        print(f"Visualization saved to: {result.html_path}")
        print(f"Open this file in a web browser to view the results")

if __name__ == "__main__":
    main()
