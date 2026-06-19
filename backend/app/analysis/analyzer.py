"""Content Analysis Module - Main Analyzer"""
from typing import List, Dict, Any
from loguru import logger
import json

from app.analysis.filler_detector import FillerWordDetector
from app.analysis.stutter_detector import StutterDetector
from app.analysis.pause_optimizer import PauseOptimizer
from app.analysis.repetition_detector import RepetitionDetector
from app.core.config import settings

class ContentAnalyzer:
    """Main content analysis orchestrator"""
    
    def __init__(self):
        logger.info("Initializing ContentAnalyzer")
        self.filler_detector = FillerWordDetector()
        self.stutter_detector = StutterDetector()
        self.pause_optimizer = PauseOptimizer(
            pause_threshold=settings.PAUSE_THRESHOLD,
            pause_target=settings.PAUSE_TARGET,
        )
        self.repetition_detector = RepetitionDetector()
        logger.info("ContentAnalyzer initialized")
    
    def analyze_content(
        self,
        sentences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis
        
        Args:
            sentences: List of sentences with word-level data
        
        Returns:
            Analysis result with all detected issues
        """
        try:
            logger.info("Starting comprehensive content analysis")
            
            # Detect filler words
            filler_words = self.filler_detector.detect_filler_words(sentences)
            
            # Detect stutters
            stutters = self.stutter_detector.detect_stutters(sentences)
            
            # Detect pauses
            pauses = self.pause_optimizer.detect_pauses(sentences)
            pause_optimizations = self.pause_optimizer.calculate_optimization(pauses)
            
            # Detect repetitions
            repetitions = self.repetition_detector.detect_repetitions(
                sentences,
                window_seconds=settings.REPETITION_WINDOW,
            )
            
            result = {
                "analysis_type": "comprehensive",
                "timestamp": None,
                "sentence_count": len(sentences),
                "total_duration": sentences[-1]["end"] if sentences else 0,
                
                "filler_words": {
                    "count": len(filler_words),
                    "items": filler_words,
                },
                
                "stutters": {
                    "count": len(stutters),
                    "items": stutters,
                },
                
                "pauses": {
                    "total_detected": len(pauses),
                    "need_optimization": len(pause_optimizations),
                    "items": pauses,
                    "optimizations": pause_optimizations,
                },
                
                "repetitions": {
                    "count": len(repetitions),
                    "items": repetitions,
                },
                
                "summary": {
                    "estimated_removable_words": len(filler_words),
                    "estimated_stutter_removals": sum([s["removable_count"] for s in stutters]),
                    "estimated_pause_reductions": len(pause_optimizations),
                    "estimated_duplicate_removals": len(repetitions),
                    "total_potential_edits": len(filler_words) + len(stutters) + len(pause_optimizations) + len(repetitions),
                },
            }
            
            logger.info(f"Analysis complete: {result['summary']['total_potential_edits']} potential edits found")
            return result
        
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            raise
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: str,
    ):
        """
        Save analysis result to JSON file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Analysis saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")
            raise
