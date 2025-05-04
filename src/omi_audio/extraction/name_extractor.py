#!/usr/bin/env python3
"""
Speaker Name Extraction Module
-----------------------------
Extracts potential speaker names from transcript text using NLP techniques.
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union

# Import spaCy if available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NameExtractor:
    """Extract potential speaker names from transcript text"""
    
    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialize the name extractor
        
        Args:
            language_model: spaCy language model to use
        """
        self.nlp = None
        
        # Check if spaCy is available
        if not SPACY_AVAILABLE:
            logger.warning("spaCy is not installed. Name extraction will be limited.")
            return
        
        # Try to load the model
        try:
            self.nlp = spacy.load(language_model)
            logger.info(f"Loaded spaCy model: {language_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model {language_model}: {str(e)}")
            logger.warning("Try installing it with: python -m spacy download en_core_web_sm")
    
    def extract_names(self, transcript_text: str) -> List[str]:
        """
        Extract potential speaker names from transcript text
        
        Args:
            transcript_text: Full transcript text
            
        Returns:
            List of extracted names
        """
        names = set()
        
        # Use spaCy for named entity recognition if available
        if self.nlp:
            doc = self.nlp(transcript_text)
            
            # Extract person names
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Clean up the name
                    name = self._clean_name(ent.text)
                    if name:
                        names.add(name)
            
            # Look for potential name patterns with titles
            name_patterns = self._find_name_patterns(transcript_text)
            names.update(name_patterns)
        else:
            # Fallback to simple regex-based extraction
            logger.info("Using regex-based name extraction (limited accuracy)")
            
            # Find capitalized words that might be names
            potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', transcript_text)
            
            # Filter out common non-name capitalized words
            for name in potential_names:
                if len(name) > 2 and not self._is_common_word(name):
                    names.add(name)
        
        return list(names)
    
    def extract_names_by_speaker(
        self, 
        aligned_segments: List[Dict],
        extract_from_text: bool = True
    ) -> Dict[str, List[str]]:
        """
        Extract potential names for each speaker based on context
        
        Args:
            aligned_segments: List of aligned speaker segments
            extract_from_text: Whether to extract names from the text
            
        Returns:
            Dictionary mapping speaker IDs to potential names
        """
        # Initialize result
        speaker_names = {}
        all_speakers = set()
        
        # Get all speaker IDs
        for segment in aligned_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            all_speakers.add(speaker)
            if speaker not in speaker_names:
                speaker_names[speaker] = []
        
        # Skip if no speaker segments
        if not aligned_segments:
            return speaker_names
        
        # Process each segment
        for i, segment in enumerate(aligned_segments):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            
            # Skip if no text
            if not text:
                continue
            
            # Check for introduction patterns
            self._check_introductions(segment, aligned_segments, i, speaker_names)
            
            # Extract names from the text if requested
            if extract_from_text and self.nlp:
                names = self.extract_names(text)
                
                # Add extracted names to the speaker
                for name in names:
                    if name not in speaker_names[speaker]:
                        speaker_names[speaker].append(name)
        
        # Filter and sort names
        for speaker in speaker_names:
            if speaker_names[speaker]:
                # Remove duplicates while preserving order
                seen = set()
                speaker_names[speaker] = [
                    name for name in speaker_names[speaker] 
                    if not (name in seen or seen.add(name))
                ]
        
        return speaker_names
    
    def _check_introductions(
        self, 
        segment: Dict, 
        all_segments: List[Dict],
        index: int,
        speaker_names: Dict[str, List[str]]
    ):
        """
        Check for introduction patterns in segments
        
        Args:
            segment: Current segment
            all_segments: All aligned segments
            index: Index of current segment
            speaker_names: Dictionary of speaker names to update
        """
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").lower()
        
        # Patterns indicating self-introduction
        intro_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"this is (\w+)",
            r"(\w+) here"
        ]
        
        # Check for self-introductions
        for pattern in intro_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = self._clean_name(match)
                if name and name not in speaker_names[speaker]:
                    logger.debug(f"Found self-introduction: {speaker} -> {name}")
                    speaker_names[speaker].insert(0, name)  # Add with high priority
        
        # Check for third-person introductions in adjacent segments
        if index > 0:
            prev_segment = all_segments[index-1]
            prev_speaker = prev_segment.get("speaker", "UNKNOWN")
            prev_text = prev_segment.get("text", "").lower()
            
            # Patterns for introducing someone else
            third_person_patterns = [
                r"this is (\w+)",
                r"that was (\w+)",
                r"next is (\w+)",
                r"(\w+) will",
                r"from (\w+)"
            ]
            
            for pattern in third_person_patterns:
                matches = re.findall(pattern, prev_text, re.IGNORECASE)
                for match in matches:
                    name = self._clean_name(match)
                    if name and name not in speaker_names[speaker]:
                        logger.debug(f"Found third-person introduction: {prev_speaker} introduces {speaker} as {name}")
                        speaker_names[speaker].insert(0, name)  # Add with high priority
    
    def _find_name_patterns(self, text: str) -> Set[str]:
        """
        Find name patterns with titles/honorifics
        
        Args:
            text: Text to search
            
        Returns:
            Set of extracted names
        """
        names = set()
        
        # Common honorifics and titles
        honorifics = [
            "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Professor",
            "Sir", "Dame", "Lord", "Lady", "Rev", "Reverend", "Hon",
            "Honorable", "Cllr", "Councillor", "Sen", "Senator", "Rep",
            "Representative", "Gov", "Governor", "Ambassador", "Judge"
        ]
        
        # Pattern: honorific + name
        for honorific in honorifics:
            pattern = fr"\b{honorific}\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
            matches = re.findall(pattern, text)
            for match in matches:
                name = self._clean_name(match)
                if name:
                    names.add(name)
        
        return names
    
    def _clean_name(self, name: str) -> Optional[str]:
        """
        Clean up extracted name
        
        Args:
            name: Raw extracted name
            
        Returns:
            Cleaned name or None if invalid
        """
        if not name:
            return None
        
        # Remove non-name parts
        name = re.sub(r'[^\w\s\'-]', '', name).strip()
        
        # Skip if too short
        if len(name) < 2:
            return None
        
        # Skip common non-name words
        if self._is_common_word(name):
            return None
        
        return name
    
    def _is_common_word(self, word: str) -> bool:
        """
        Check if a word is a common non-name word
        
        Args:
            word: Word to check
            
        Returns:
            True if common word, False otherwise
        """
        # Common words that might be capitalized but aren't names
        common_words = {
            "The", "This", "That", "There", "Their", "They", "Then", "These",
            "Those", "Today", "Tomorrow", "Yesterday", "Here", "Hello", "Hi",
            "Yes", "No", "Maybe", "Okay", "Sure", "Thanks", "Thank", "Please",
            "Sorry", "Excuse", "Good", "Great", "Nice", "Well", "Very", "Really",
            "Just", "Now", "Next", "Last", "First", "Second", "Third", "One",
            "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        }
        
        return word in common_words


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python name_extractor.py <transcript_text>")
        sys.exit(1)
    
    extractor = NameExtractor()
    
    # Extract names from text
    text = sys.argv[1]
    names = extractor.extract_names(text)
    
    print("Extracted names:")
    for name in names:
        print(f"- {name}")
