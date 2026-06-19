"""Content Analysis Module - Filler Words Detection"""
from typing import List, Dict, Any
from loguru import logger

class FillerWordDetector:
    """Detect and analyze filler words in Chinese speech"""
    
    # Filler words dictionary
    FILLER_WORDS = {
        "呃": {"pinyin": "e", "type": "filler"},
        "额": {"pinyin": "e", "type": "filler"},
        "嗯": {"pinyin": "en", "type": "filler"},
        "啊": {"pinyin": "a", "type": "filler"},
        "那个": {"pinyin": "nage", "type": "filler"},
        "这个": {"pinyin": "zhege", "type": "filler"},
        "然后": {"pinyin": "ranhou", "type": "connector"},
        "就是": {"pinyin": "jiushi", "type": "connector"},
        "那么": {"pinyin": "name", "type": "connector"},
    }
    
    def __init__(self):
        logger.info("FillerWordDetector initialized")
    
    def detect_filler_words(
        self,
        sentences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect filler words in sentences
        
        Args:
            sentences: List of sentences with word data
        
        Returns:
            List of filler word occurrences
        """
        try:
            filler_occurrences = []
            
            for sentence in sentences:
                words = sentence.get("words", [])
                filler_count = 0
                
                for idx, word in enumerate(words):
                    word_text = word["word"].strip()
                    
                    # Check if word is a filler word
                    if word_text in self.FILLER_WORDS:
                        filler_info = self.FILLER_WORDS[word_text]
                        
                        # Rule: Max 1 filler word per sentence
                        if filler_count == 0:
                            filler_occurrences.append({
                                "sentence_id": sentence["id"],
                                "word_id": word["id"],
                                "word": word_text,
                                "start": word["start"],
                                "end": word["end"],
                                "duration": word["end"] - word["start"],
                                "type": filler_info["type"],
                                "context_before": words[idx - 1]["word"] if idx > 0 else None,
                                "context_after": words[idx + 1]["word"] if idx < len(words) - 1 else None,
                                "removable": True,
                                "confidence": 0.95,
                            })
                            filler_count += 1
            
            logger.info(f"Detected {len(filler_occurrences)} filler words")
            return filler_occurrences
        
        except Exception as e:
            logger.error(f"Filler word detection failed: {str(e)}")
            raise
    
    def is_filler_word(self, word: str) -> bool:
        """Check if word is a filler word"""
        return word.strip() in self.FILLER_WORDS
    
    def get_filler_words_list(self) -> List[str]:
        """Get list of all known filler words"""
        return list(self.FILLER_WORDS.keys())
