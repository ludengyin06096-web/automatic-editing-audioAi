"""Content Analysis Module - Stutter Detection"""
from typing import List, Dict, Any
from loguru import logger
import re

class StutterDetector:
    """Detect and analyze stutters in speech"""
    
    def __init__(self):
        logger.info("StutterDetector initialized")
    
    def detect_stutters(
        self,
        sentences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect stutters (repeated syllables/words)
        
        Args:
            sentences: List of sentences with word data
        
        Returns:
            List of stutter occurrences
        """
        try:
            stutter_occurrences = []
            
            for sentence in sentences:
                words = sentence.get("words", [])
                
                # Check for consecutive repeated words
                for idx in range(len(words) - 1):
                    current_word = words[idx]["word"].strip()
                    next_word = words[idx + 1]["word"].strip()
                    
                    # Skip punctuation and spaces
                    if not current_word or not next_word:
                        continue
                    
                    # Check for repetition (e.g., "我我我", "那那那")
                    if self._is_stutter(current_word, next_word):
                        # Find all consecutive repetitions
                        repeat_count = 2
                        check_idx = idx + 2
                        while check_idx < len(words):
                            if words[check_idx]["word"].strip() == current_word:
                                repeat_count += 1
                                check_idx += 1
                            else:
                                break
                        
                        # Only flag if more than 2 consecutive repetitions
                        if repeat_count >= 2:
                            end_idx = idx + repeat_count - 1
                            repeat_words = words[idx:end_idx + 1]
                            
                            stutter_occurrences.append({
                                "sentence_id": sentence["id"],
                                "start_word_id": words[idx]["id"],
                                "end_word_id": words[end_idx]["id"],
                                "word": current_word,
                                "repeat_count": repeat_count,
                                "start": words[idx]["start"],
                                "end": words[end_idx]["end"],
                                "duration": words[end_idx]["end"] - words[idx]["start"],
                                "pattern": "".join([w["word"] for w in repeat_words]),
                                "removable_count": repeat_count - 1,
                                "confidence": 0.90,
                            })
            
            logger.info(f"Detected {len(stutter_occurrences)} stutters")
            return stutter_occurrences
        
        except Exception as e:
            logger.error(f"Stutter detection failed: {str(e)}")
            raise
    
    @staticmethod
    def _is_stutter(word1: str, word2: str) -> bool:
        """Check if two consecutive words are a stutter"""
        # Normalize spaces and punctuation
        w1 = word1.strip()
        w2 = word2.strip()
        
        return w1 == w2 and len(w1) > 0
