#!/usr/bin/env python3
"""
Speaker Embedding Extractor
--------------------------
Extract speaker embeddings using SpeechBrain's ECAPA-TDNN model.

This module handles the extraction of high-quality speaker embeddings
from audio segments, which are critical for accurate speaker identification
and profile matching across multiple recordings.
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import torchaudio
import librosa

try:
    # SpeechBrain 1.0+ uses the inference module path
    from speechbrain.inference import EncoderClassifier
except ImportError:
    # Fallback for older SpeechBrain versions
    from speechbrain.pretrained import EncoderClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model path for caching
DEFAULT_MODEL_PATH = Path("models/ecapa-tdnn")
DEFAULT_MODEL_PATH.mkdir(exist_ok=True, parents=True)


class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings using SpeechBrain's ECAPA-TDNN model"""
    
    def __init__(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize speaker embedding extractor
        
        Args:
            model_path: Path to pretrained model
            cache_dir: Directory to cache model files
        """
        self.model_path = model_path or "speechbrain/spkrec-ecapa-voxceleb"
        self.cache_dir = cache_dir or str(DEFAULT_MODEL_PATH)
        
        # Load model (will download if needed)
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_path,
            savedir=self.cache_dir,
            run_opts={"device": "cpu"}
        )
        
        logger.info(f"Loaded ECAPA-TDNN speaker embedding model on {self.classifier.device}")
    
    def extract_from_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract embedding from entire audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding as numpy array
        """
        try:
            signal = self.classifier.load_audio(str(audio_path))
            embedding = self.classifier.encode_batch(signal)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(192)  # Default ECAPA-TDNN dimension
    
    def extract_from_segment(self, audio_path: Union[str, Path], 
                             start_time: float, end_time: float) -> np.ndarray:
        """
        Extract embedding from audio segment
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Speaker embedding as numpy array
        """
        try:
            # Load full audio
            signal, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert times to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment
            segment = signal[:, start_sample:end_sample]
            
            # If segment is too short, pad it
            if segment.shape[1] < sample_rate:
                pad_length = sample_rate - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_length))
            
            # Extract embedding
            embedding = self.classifier.encode_batch(segment)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting segment embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(192)  # Default ECAPA-TDNN dimension
    
    def extract_from_segments(self, audio_path: Union[str, Path], 
                              segments: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for multiple segments
        
        Args:
            audio_path: Path to audio file
            segments: List of segment dictionaries with speaker_id, start_time, end_time
            
        Returns:
            Dictionary mapping speaker IDs to embeddings
        """
        # Group segments by speaker to extract one embedding per speaker
        speakers = {}
        for segment in segments:
            speaker_id = segment.get("speaker_id") or segment.get("speaker")
            if not speaker_id:
                continue
                
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(segment)
        
        # Extract embeddings for each speaker
        embeddings = {}
        for speaker_id, speaker_segments in speakers.items():
            # Sort segments by duration (descending)
            sorted_segments = sorted(
                speaker_segments, 
                key=lambda x: x.get("end_time", 0) - x.get("start_time", 0),
                reverse=True
            )
            
            # Use the longest segment for better quality
            longest_segment = sorted_segments[0]
            start_time = longest_segment.get("start_time", 0)
            end_time = longest_segment.get("end_time", 0)
            
            # Extract embedding
            embedding = self.extract_from_segment(audio_path, start_time, end_time)
            embeddings[speaker_id] = embedding
        
        logger.info(f"Extracted embeddings for {len(embeddings)} speakers")
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Ensure value is in range [0, 1]
        return max(0.0, min(float(similarity), 1.0))


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    extractor = SpeakerEmbeddingExtractor()
    
    # Extract embedding from full file
    embedding = extractor.extract_from_file(audio_path)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Extract embedding from first 5 seconds
    segment_embedding = extractor.extract_from_segment(audio_path, 0, 5)
    print(f"Segment embedding shape: {segment_embedding.shape}")
    
    # Compare similarity
    similarity = extractor.compute_similarity(embedding, segment_embedding)
    print(f"Similarity between full and segment: {similarity:.4f}")