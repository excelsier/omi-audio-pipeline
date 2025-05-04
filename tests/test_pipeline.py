#!/usr/bin/env python3
"""
Basic tests for the OMI Audio Pipeline
"""
import os
import sys
import unittest
from pathlib import Path

# Add the src directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi_audio.pipeline.complete_pipeline import CompletePipeline
from omi_audio.diarization.robust_diarizer import RobustDirectPyAnnoteDiarizer


class TestPipeline(unittest.TestCase):
    """Test the complete audio pipeline functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Skip tests if API keys are not set
        if not os.environ.get("PYANNOTE_API_KEY"):
            self.skipTest("PYANNOTE_API_KEY environment variable not set")
        
        # Initialize pipeline with test configuration
        self.pipeline = CompletePipeline(
            min_speakers=1,
            max_speakers=3,
            clustering_threshold=0.7
        )
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.diarizer)
        self.assertIsNotNone(self.pipeline.transcriber)
        self.assertIsNotNone(self.pipeline.embedding_extractor)
    
    def test_diarizer_parameters(self):
        """Test that diarizer parameters are set correctly"""
        diarizer = self.pipeline.diarizer
        self.assertIsInstance(diarizer, RobustDirectPyAnnoteDiarizer)
        self.assertEqual(diarizer.min_speakers, 1)
        self.assertEqual(diarizer.max_speakers, 3)
        self.assertAlmostEqual(diarizer.clustering_threshold, 0.7)


if __name__ == "__main__":
    unittest.main()
