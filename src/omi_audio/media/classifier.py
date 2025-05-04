#!/usr/bin/env python3
"""
Media Content Classifier
-----------------------
Analyzes audio content and transcriptions to determine media type and topics.

Uses transformer models to classify content based on the transcript.
"""
import logging
from typing import Dict, List, Optional, Union, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Media classification will be disabled.")


class MediaClassifier:
    """Classifies audio content based on transcript text"""
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[str] = None
    ):
        """
        Initialize the media classifier
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
        # Skip if transformers not available
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Media classification disabled (transformers not installed)")
            return
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            logger.info(f"Loading classification model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading classification model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def classify_text(
        self, 
        text: str, 
        max_length: int = 512,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Classify text content
        
        Args:
            text: Text to classify
            max_length: Maximum token length
            threshold: Confidence threshold for classification
            
        Returns:
            Classification results
        """
        # Check if model is available
        if not self.model or not self.tokenizer:
            logger.warning("Classification model not available")
            return {"error": "Model not available"}
        
        try:
            # Truncate text if too long
            if len(text) > max_length * 10:  # Rough character estimate
                logger.info(f"Truncating text from {len(text)} characters")
                text = text[:max_length * 10]
            
            # Tokenize and classify
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get prediction
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            probabilities = probabilities.cpu().numpy()
            
            # Get class labels if available
            if hasattr(self.model.config, "id2label"):
                labels = self.model.config.id2label
            else:
                labels = {i: f"Class {i}" for i in range(len(probabilities))}
            
            # Format results
            results = []
            for i, prob in enumerate(probabilities):
                if prob >= threshold:
                    results.append({
                        "label": labels[i],
                        "score": float(prob)
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "classifications": results,
                "model": self.model_name
            }
        
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {"error": str(e)}
    
    def classify_transcript(
        self, 
        transcript_segments: List[Dict[str, Any]], 
        chunk_size: int = 5,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Classify transcript content
        
        Args:
            transcript_segments: List of transcript segments
            chunk_size: Number of segments to combine for each classification
            threshold: Confidence threshold for classification
            
        Returns:
            Classification results
        """
        # Check if model is available
        if not self.model or not self.tokenizer:
            logger.warning("Classification model not available")
            return {"error": "Model not available"}
        
        # Extract text from segments
        all_text = " ".join([seg.get("text", "") for seg in transcript_segments if "text" in seg])
        
        # Classify full text
        classification = self.classify_text(all_text, threshold=threshold)
        
        # Add metadata
        classification["segments_count"] = len(transcript_segments)
        classification["text_length"] = len(all_text)
        
        return classification
    
    def get_categories(self, classifications: List[Dict[str, Any]]) -> List[str]:
        """
        Extract category names from classifications
        
        Args:
            classifications: Classification results
            
        Returns:
            List of category names
        """
        return [c["label"] for c in classifications]
    
    def get_main_category(self, classifications: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get main category from classifications
        
        Args:
            classifications: Classification results
            
        Returns:
            Main category name or None
        """
        if not classifications:
            return None
        return classifications[0]["label"]


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <text_to_classify>")
        sys.exit(1)
    
    classifier = MediaClassifier()
    
    # Classify text
    text = sys.argv[1]
    result = classifier.classify_text(text)
    
    print(json.dumps(result, indent=2))
