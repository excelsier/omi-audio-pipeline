#!/usr/bin/env python3
"""
Complete Audio Processing Pipeline
---------------------------------
Integrates diarization, transcription, and speaker identification components.

This module provides a comprehensive end-to-end pipeline for processing audio files,
including speaker diarization, transcription, and speaker identification.
"""
import os
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Import pipeline components
from omi_audio.diarization.robust_diarizer import RobustDirectPyAnnoteDiarizer
from omi_audio.transcription.elevenlabs import ElevenLabsTranscriptionService, TranscriptionSegment
from omi_audio.speaker.embeddings import SpeakerEmbeddingExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ProcessingResult:
    """Container for the complete audio processing result"""
    output_id: str
    diarization: Dict[str, Any]
    transcription: Dict[str, Any]
    aligned_segments: List[Dict[str, Any]] = None
    speaker_profiles: Dict[str, Any] = None
    html_path: Optional[str] = None
    json_path: Optional[str] = None
    audio_path: Optional[str] = None
    media_classification: Dict[str, Any] = None


class CompletePipeline:
    """End-to-end audio processing pipeline"""
    
    def __init__(
        self,
        diarizer=None,
        transcriber=None,
        embedding_extractor=None,
        results_dir=None,
        min_speakers=2,
        max_speakers=8,
        clustering_threshold=0.7
    ):
        """
        Initialize the complete pipeline
        
        Args:
            diarizer: Diarizer instance (or None to create default)
            transcriber: Transcription service (or None to create default)
            embedding_extractor: Speaker embedding extractor (or None to create default)
            results_dir: Directory to save results
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            clustering_threshold: Threshold for speaker clustering
        """
        # Initialize component services
        self.diarizer = diarizer or RobustDirectPyAnnoteDiarizer(
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold
        )
        
        self.transcriber = transcriber or ElevenLabsTranscriptionService()
        self.embedding_extractor = embedding_extractor or SpeakerEmbeddingExtractor()
        
        # Set up results directory
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Complete pipeline initialized successfully")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        output_id: Optional[str] = None,
        languages: Optional[List[str]] = None,
        save_result: bool = True,
        generate_visualization: bool = True,
        # New parameters for speaker profiles
        update_speaker_profiles: bool = True,
        recording_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process audio through the complete pipeline, combining diarization and transcription
        
        Args:
            audio_path: Path to audio file
            output_id: Custom identifier for output files
            languages: List of language codes for transcription
            save_result: Whether to save results to file
            generate_visualization: Whether to generate HTML visualization
            update_speaker_profiles: Whether to update speaker profiles
            recording_id: Identifier for the recording
            
        Returns:
            ProcessingResult object containing all results
        """
        start_time = time.time()
        
        # Ensure audio path is a Path object
        audio_path = Path(audio_path)
        
        # Generate output ID if not provided
        if not output_id:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_id = f"{timestamp}_{audio_path.stem}"
        
        # Generate recording ID if not provided
        if not recording_id:
            recording_id = output_id
        
        # Step 1: Perform speaker diarization
        logger.info(f"Starting diarization for: {audio_path}")
        diarization_result = self.diarizer.process_audio(audio_path, output_id=output_id)
        
        # Step 2: Perform transcription
        logger.info(f"Starting transcription for: {audio_path}")
        transcription_result = self.transcriber.transcribe(audio_path, languages=languages)
        
        # Step 3: Align speakers with transcript
        logger.info("Aligning speakers with transcript")
        diarization_segments = diarization_result.get("segments", [])
        transcription_segments = transcription_result.get("segments", [])
        
        # Get aligned segments - combine diarization with transcription
        aligned_segments = self._align_speakers_with_transcript(
            diarization_segments, 
            transcription_segments
        )
        
        # Step 4: Process speaker profiles (optional)
        speaker_profiles = {}
        if update_speaker_profiles:
            logger.info("Extracting speaker embeddings")
            speaker_embeddings = self.embedding_extractor.extract_from_segments(
                audio_path, 
                diarization_segments
            )
            
            # Create basic speaker profiles
            for speaker_id, embedding in speaker_embeddings.items():
                # Find all segments for this speaker
                segments = [s for s in aligned_segments if s.get("speaker") == speaker_id]
                
                # Calculate total speaking time
                total_duration = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
                
                # Create profile
                speaker_profiles[speaker_id] = {
                    "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                    "recordings": [recording_id],
                    "total_duration": total_duration,
                    "segments_count": len(segments),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # Step 5: Create result
        result = ProcessingResult(
            output_id=output_id,
            diarization=diarization_result,
            transcription=transcription_result,
            aligned_segments=aligned_segments,
            speaker_profiles=speaker_profiles,
            audio_path=str(audio_path)
        )
        
        # Step 6: Save result to files
        if save_result:
            # Save JSON result
            json_path = self.results_dir / f"{output_id}_complete.json"
            with open(json_path, "w") as f:
                # Create serializable version of the result
                serializable_result = {
                    "output_id": output_id,
                    "diarization": diarization_result,
                    "transcription": transcription_result,
                    "aligned_segments": aligned_segments,
                    "speaker_profiles": speaker_profiles,
                    "processing_time": time.time() - start_time,
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                json.dump(serializable_result, f, indent=2)
            
            result.json_path = str(json_path)
            logger.info(f"Results saved to: {json_path}")
            
            # Generate HTML visualization if requested
            if generate_visualization:
                html_path = self.results_dir / f"{output_id}_visualization.html"
                self._generate_visualization(result, html_path)
                result.html_path = str(html_path)
                logger.info(f"Visualization saved to: {html_path}")
        
        # Log processing stats
        processing_time = time.time() - start_time
        audio_duration = transcription_result.get("duration", 0)
        processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0
        
        logger.info(f"Complete processing finished in {processing_time:.2f}s for {audio_duration:.2f}s audio "
                   f"(processing ratio: {processing_ratio:.2f}x)")
        logger.info(f"Found {diarization_result.get('num_speakers', 0)} speakers and "
                   f"{transcription_result.get('num_segments', 0)} transcript segments")
        
        return result
    
    def _align_speakers_with_transcript(
        self,
        diarization_segments: List[Dict[str, Any]],
        transcription_segments: List[TranscriptionSegment]
    ) -> List[Dict[str, Any]]:
        """
        Align speakers with transcribed text based on time overlap
        
        Args:
            diarization_segments: Diarization segments with speaker info
            transcription_segments: Transcription segments with text
            
        Returns:
            List of aligned segments with speaker and text
        """
        aligned_segments = []
        
        for trans_segment in transcription_segments:
            trans_start = trans_segment.start_time
            trans_end = trans_segment.end_time
            
            # Find the speaker with maximum overlap for this segment
            max_overlap = 0
            best_speaker = None
            
            for diar_segment in diarization_segments:
                diar_start = diar_segment.get("start_time", 0)
                diar_end = diar_segment.get("end_time", 0)
                speaker = diar_segment.get("speaker", "UNKNOWN")
                
                # Calculate overlap between segments
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # Update best speaker if this overlap is greater
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
            
            # Create aligned segment
            aligned_segments.append({
                "speaker": best_speaker or "UNKNOWN",
                "start": trans_start,
                "end": trans_end,
                "text": trans_segment.text,
                "words": trans_segment.words
            })
        
        return aligned_segments
    
    def _generate_visualization(self, result: ProcessingResult, output_path: Union[str, Path]):
        """
        Generate HTML visualization of the processing result
        
        Args:
            result: Processing result
            output_path: Output HTML file path
        """
        # Get data for visualization
        diarization = result.diarization
        transcription = result.transcription
        aligned_segments = result.aligned_segments
        
        # Generate unique colors for each speaker
        speakers = diarization.get("speakers", {})
        num_speakers = len(speakers)
        
        # Create color map
        colors = self._generate_speaker_colors(num_speakers)
        speaker_colors = {}
        
        for i, speaker_id in enumerate(speakers.keys()):
            speaker_colors[speaker_id] = colors[i % len(colors)]
        
        # Create HTML content
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Audio Processing Result</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #333; }
                .segment { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .speaker-label { font-weight: bold; display: inline-block; min-width: 100px; }
                .timestamp { color: #666; font-size: 0.9em; margin-left: 10px; }
                .transcript { margin-top: 5px; }
                .speakers-summary { display: flex; flex-wrap: wrap; margin: 20px 0; }
                .speaker-box { padding: 10px; margin: 5px; border-radius: 5px; flex: 0 0 calc(33% - 20px); }
                .timeline { position: relative; height: 80px; background: #f0f0f0; margin: 20px 0; overflow: hidden; }
                .timeline-segment { position: absolute; height: 100%; top: 0; }
                .audio-player { width: 100%; margin: 20px 0; }
                .visualization-tabs { display: flex; margin-bottom: 20px; }
                .tab { padding: 10px 20px; cursor: pointer; background: #f0f0f0; margin-right: 5px; border-radius: 5px 5px 0 0; }
                .tab.active { background: #007BFF; color: white; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
            </style>
        </head>
        <body>
            <h1>Audio Processing Result</h1>
            <div class="summary">
                <p><strong>File:</strong> FILENAME</p>
                <p><strong>Duration:</strong> DURATION seconds</p>
                <p><strong>Speakers:</strong> NUM_SPEAKERS</p>
                <p><strong>Processed at:</strong> PROCESSED_AT</p>
            </div>
            
            <div class="visualization-tabs">
                <div class="tab active" data-tab="transcript">Transcript</div>
                <div class="tab" data-tab="speakers">Speakers</div>
                <div class="tab" data-tab="timeline">Timeline</div>
            </div>
            
            <div id="transcript" class="tab-content active">
                <h2>Aligned Transcript</h2>
                <div class="aligned-transcript">
                    ALIGNED_SEGMENTS
                </div>
            </div>
            
            <div id="speakers" class="tab-content">
                <h2>Speakers Summary</h2>
                <div class="speakers-summary">
                    SPEAKERS_SUMMARY
                </div>
            </div>
            
            <div id="timeline" class="tab-content">
                <h2>Timeline Visualization</h2>
                <div class="timeline">
                    TIMELINE_SEGMENTS
                </div>
            </div>
            
            <script>
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', () => {
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        
                        tab.classList.add('active');
                        document.getElementById(tab.dataset.tab).classList.add('active');
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Replace placeholders with actual content
        html = html.replace("FILENAME", str(result.audio_path))
        html = html.replace("DURATION", str(round(transcription.get("duration", 0), 2)))
        html = html.replace("NUM_SPEAKERS", str(diarization.get("num_speakers", 0)))
        html = html.replace("PROCESSED_AT", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate aligned segments HTML
        aligned_segments_html = ""
        for segment in aligned_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Get color for this speaker
            color = speaker_colors.get(speaker, "#cccccc")
            
            # Format timestamp
            timestamp = f"{self._format_timestamp(start)} - {self._format_timestamp(end)}"
            
            # Create segment HTML
            segment_html = f"""
            <div class="segment" style="background-color: {color}20; border-left: 5px solid {color};">
                <span class="speaker-label">{speaker}</span>
                <span class="timestamp">{timestamp}</span>
                <div class="transcript">{text}</div>
            </div>
            """
            
            aligned_segments_html += segment_html
        
        html = html.replace("ALIGNED_SEGMENTS", aligned_segments_html)
        
        # Generate speakers summary HTML
        speakers_summary_html = ""
        for speaker_id, speaker_data in speakers.items():
            color = speaker_colors.get(speaker_id, "#cccccc")
            total_duration = speaker_data.get("total_duration", 0)
            
            # Calculate percentage of speaking time
            percentage = 0
            if transcription.get("duration", 0) > 0:
                percentage = (total_duration / transcription.get("duration", 0)) * 100
            
            # Create speaker box HTML
            speaker_html = f"""
            <div class="speaker-box" style="background-color: {color}20; border-left: 5px solid {color};">
                <strong>{speaker_id}</strong>
                <div>Speaking time: {total_duration:.2f}s ({percentage:.1f}%)</div>
                <div>Segments: {len(speaker_data.get("segments", []))}</div>
            </div>
            """
            
            speakers_summary_html += speaker_html
        
        html = html.replace("SPEAKERS_SUMMARY", speakers_summary_html)
        
        # Generate timeline visualization
        timeline_html = ""
        duration = transcription.get("duration", 0)
        
        if duration > 0:
            for segment in diarization_segments:
                speaker = segment.get("speaker", "UNKNOWN")
                start = segment.get("start_time", 0)
                end = segment.get("end_time", 0)
                
                # Calculate position and width as percentages
                left_percent = (start / duration) * 100
                width_percent = ((end - start) / duration) * 100
                
                # Get color for this speaker
                color = speaker_colors.get(speaker, "#cccccc")
                
                # Create timeline segment HTML
                segment_html = f"""
                <div class="timeline-segment" style="left: {left_percent}%; width: {width_percent}%; background-color: {color};">
                    <div style="padding: 5px; font-size: 0.8em; color: white;">{speaker}</div>
                </div>
                """
                
                timeline_html += segment_html
        
        html = html.replace("TIMELINE_SEGMENTS", timeline_html)
        
        # Write HTML to file
        with open(output_path, "w") as f:
            f.write(html)
    
    def _generate_speaker_colors(self, num_colors: int) -> List[str]:
        """
        Generate distinct colors for speakers
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of hex color codes
        """
        # Predefined colors for better visual distinction
        base_colors = [
            "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
            "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395",
            "#994499", "#22AA99", "#AAAA11", "#6633CC", "#E67300",
            "#8B0707", "#329262", "#5574A6", "#3B3EAC"
        ]
        
        # If we need more colors than predefined, generate additional ones
        if num_colors <= len(base_colors):
            return base_colors[:num_colors]
        else:
            # Generate additional colors using HSV color space
            import colorsys
            
            colors = base_colors.copy()
            remaining = num_colors - len(base_colors)
            
            for i in range(remaining):
                # Generate evenly spaced hues
                h = i / remaining
                # Fixed saturation and value for vibrant colors
                s = 0.8
                v = 0.9
                
                # Convert HSV to RGB
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                
                # Convert RGB to hex
                hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                colors.append(hex_color)
            
            return colors
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as MM:SS.mmm
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds / 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:05.2f}"


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio with complete pipeline")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=8, help="Maximum number of speakers")
    parser.add_argument("--languages", nargs="+", help="Languages for transcription")
    
    args = parser.parse_args()
    
    pipeline = CompletePipeline(
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    result = pipeline.process_audio(
        args.audio_path,
        languages=args.languages
    )
    
    print(f"Processing complete!")
    print(f"Results saved to: {result.json_path}")
    if result.html_path:
        print(f"Visualization saved to: {result.html_path}")
