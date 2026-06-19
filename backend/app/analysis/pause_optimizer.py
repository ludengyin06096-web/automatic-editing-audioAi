"""Content Analysis Module - Pause Detection and Optimization"""
from typing import List, Dict, Any
from loguru import logger

class PauseOptimizer:
    """Detect and optimize pauses in speech"""
    
    def __init__(self, pause_threshold: float = 0.8, pause_target: float = 0.65):
        """
        Initialize PauseOptimizer
        
        Args:
            pause_threshold: Threshold to detect pause (seconds)
            pause_target: Target pause duration (seconds)
        """
        self.pause_threshold = pause_threshold
        self.pause_target = pause_target
        logger.info(f"PauseOptimizer initialized (threshold: {pause_threshold}s, target: {pause_target}s)")
    
    def detect_pauses(
        self,
        sentences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect pauses between words
        
        Args:
            sentences: List of sentences with word data
        
        Returns:
            List of pause occurrences
        """
        try:
            pauses = []
            
            for sentence in sentences:
                words = sentence.get("words", [])
                
                for idx in range(len(words) - 1):
                    current_word = words[idx]
                    next_word = words[idx + 1]
                    
                    # Calculate pause duration
                    pause_duration = next_word["start"] - current_word["end"]
                    
                    # Detect significant pauses
                    if pause_duration >= self.pause_threshold:
                        pauses.append({
                            "sentence_id": sentence["id"],
                            "before_word_id": current_word["id"],
                            "after_word_id": next_word["id"],
                            "before_word": current_word["word"],
                            "after_word": next_word["word"],
                            "pause_start": current_word["end"],
                            "pause_end": next_word["start"],
                            "duration": pause_duration,
                            "is_breathing": self._is_likely_breathing(pause_duration),
                            "is_emphasis": self._is_emphasis_pause(current_word, next_word),
                            "is_sentence_end": idx == len(words) - 2,
                            "optimization_target": self.pause_target,
                            "needs_optimization": pause_duration > self.pause_target,
                        })
            
            logger.info(f"Detected {len(pauses)} pauses")
            return pauses
        
        except Exception as e:
            logger.error(f"Pause detection failed: {str(e)}")
            raise
    
    @staticmethod
    def _is_likely_breathing(duration: float) -> bool:
        """Check if pause is likely breathing (0.3-0.8s)"""
        return 0.3 <= duration <= 0.8
    
    @staticmethod
    def _is_emphasis_pause(before_word: Dict, after_word: Dict) -> bool:
        """Check if pause is for emphasis"""
        # Pause before important words (heuristic)
        emphasis_words = ["其实", "重点", "关键", "但是", "所以", "因此"]
        return after_word["word"].strip() in emphasis_words
    
    def calculate_optimization(
        self,
        pauses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Calculate pause optimization recommendations
        
        Args:
            pauses: List of detected pauses
        
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            for pause in pauses:
                # Skip breathing, emphasis, and sentence-end pauses
                if pause["is_breathing"] or pause["is_emphasis"] or pause["is_sentence_end"]:
                    continue
                
                if pause["needs_optimization"]:
                    reduction = pause["duration"] - self.pause_target
                    recommendations.append({
                        "pause_id": pause["pause_start"],
                        "current_duration": round(pause["duration"], 3),
                        "target_duration": self.pause_target,
                        "reduction_amount": round(reduction, 3),
                        "action": "compress",
                        "confidence": 0.85,
                    })
            
            logger.info(f"Generated {len(recommendations)} pause optimization recommendations")
            return recommendations
        
        except Exception as e:
            logger.error(f"Optimization calculation failed: {str(e)}")
            raise
