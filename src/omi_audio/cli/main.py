#!/usr/bin/env python3
"""
OMI Audio Pipeline CLI
----------------------
Command-line interface for the OMI Audio Pipeline
"""
import os
import sys
import argparse
import logging
from pathlib import Path

from omi_audio.pipeline.complete_pipeline import CompletePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OMI Audio Pipeline - Process audio files with speaker diarization and transcription",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "audio_path", 
        help="Path to the audio file to process"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-id", 
        help="Custom identifier for output files"
    )
    output_group.add_argument(
        "--results-dir", 
        default="results",
        help="Directory to save results"
    )
    output_group.add_argument(
        "--no-visualization", 
        action="store_true",
        help="Disable HTML visualization generation"
    )
    
    # Diarization options
    diarization_group = parser.add_argument_group("Diarization Options")
    diarization_group.add_argument(
        "--min-speakers", 
        type=int, 
        default=2,
        help="Minimum number of speakers to detect"
    )
    diarization_group.add_argument(
        "--max-speakers", 
        type=int, 
        default=8,
        help="Maximum number of speakers to detect"
    )
    diarization_group.add_argument(
        "--clustering-threshold", 
        type=float, 
        default=0.7,
        help="Threshold for speaker clustering (0.0-1.0)"
    )
    
    # Transcription options
    transcription_group = parser.add_argument_group("Transcription Options")
    transcription_group.add_argument(
        "--languages", 
        nargs="+", 
        default=["en"],
        help="Languages for transcription (ISO codes, e.g., en, fr, de)"
    )
    
    # Speaker profile options
    speaker_group = parser.add_argument_group("Speaker Profile Options")
    speaker_group.add_argument(
        "--no-speaker-profiles", 
        action="store_true",
        help="Disable speaker profile generation"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check that audio file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Check for API keys
    if not os.environ.get("PYANNOTE_API_KEY"):
        logger.error("PYANNOTE_API_KEY environment variable is not set")
        logger.error("Please set it to use the diarization functionality")
        return 1
    
    if not os.environ.get("ELEVENLABS_API_KEY"):
        logger.warning("ELEVENLABS_API_KEY environment variable is not set")
        logger.warning("Using default transcription service instead of ElevenLabs")
    
    try:
        # Initialize pipeline
        pipeline = CompletePipeline(
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            clustering_threshold=args.clustering_threshold,
            results_dir=results_dir
        )
        
        # Process audio
        result = pipeline.process_audio(
            audio_path,
            output_id=args.output_id,
            languages=args.languages,
            generate_visualization=not args.no_visualization,
            update_speaker_profiles=not args.no_speaker_profiles
        )
        
        # Print results
        logger.info("Processing completed successfully!")
        logger.info(f"Found {result.diarization.get('num_speakers', 0)} speakers in the audio")
        
        # Output file locations
        if result.json_path:
            logger.info(f"Results saved to: {result.json_path}")
        if result.html_path:
            logger.info(f"Visualization saved to: {result.html_path}")
            logger.info(f"Open this file in a web browser to view the results")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error processing audio: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
