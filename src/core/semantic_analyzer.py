"""Semantic and linguistic analysis module for audio content"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from .transcriber import TranscriptSegment, TranscriptResult


class StopPauseType(Enum):
    """Classification of pause types"""
    BREATH = "breath"              # Natural breathing pause
    THOUGHT = "thought"            # Thinking/hesitation pause
    EMOTION = "emotion"            # Emotional emphasis pause
    EMPHASIS = "emphasis"          # Deliberate pause for emphasis
    NARRATIVE = "narrative"        # Narrative structure pause
    REDUNDANT = "redundant"        # Unnecessary pause


class FillerWordCategory(Enum):
    """Filler word categories"""
    THOUGHT_MARKER = "thought_marker"  # 呃、嗯、额
    TRANSITION = "transition"          # 然后、就是、那么
    DEICTIC = "deictic"                # 这个、那个
    MEANINGLESS = "meaningless"        # Pure filler with no semantic value


@dataclass
class SemanticSegment:
    """Analyzed semantic segment"""
    segment: TranscriptSegment
    segment_type: str  # 'statement', 'question', 'emphasis', 'thought'
    semantic_importance: float  # 0.0-1.0
    emotion_markers: List[str]
    contains_filler: bool
    filler_words: List[str]
    pause_before: Optional[float]
    pause_type_before: Optional[StopPauseType]


class SemanticAnalyzer:
    """Analyze semantic structure, emotion markers, and filler words"""

    # Chinese filler words by category
    FILLER_WORDS = {
        FillerWordCategory.THOUGHT_MARKER: ["呃", "嗯", "额", "啊", "唔"],
        FillerWordCategory.TRANSITION: ["然后", "就是", "那么", "所以", "而且"],
        FillerWordCategory.DEICTIC: ["这个", "那个", "这样", "那样"],
        FillerWordCategory.MEANINGLESS: ["咦", "呀", "哎呀"],
    }

    # Emphasis markers (strong semantic importance)
    EMPHASIS_MARKERS = ["非常", "特别", "真的", "一定", "必须", "绝对", "完全", "真正"]

    # Question markers
    QUESTION_MARKERS = ["吗", "呢", "吧", "？", "?", "怎样", "为什么", "怎么"]

    # Thought/hesitation markers
    THOUGHT_MARKERS = ["好", "嗯", "呃", "让我想想", "我觉得", "似乎", "可能"]

    def __init__(self):
        """Initialize semantic analyzer"""
        logger.info("SemanticAnalyzer initialized")

    def analyze_transcript(
        self,
        result: TranscriptResult,
        min_silence_duration: float = 0.3
    ) -> List[SemanticSegment]:
        """Analyze complete transcript
        
        Args:
            result: TranscriptResult from transcriber
            min_silence_duration: Minimum pause duration to consider
            
        Returns:
            List of analyzed semantic segments
        """
        analyzed_segments = []
        
        for i, segment in enumerate(result.segments):
            # Analyze current segment
            seg_type = self._classify_segment_type(segment.text)
            importance = self._calculate_importance(segment.text)
            emotions = self._extract_emotion_markers(segment.text)
            has_filler, fillers = self._extract_fillers(segment.text)
            
            # Analyze pause before this segment
            pause_before = None
            pause_type = None
            if i > 0:
                pause_before = segment.start - result.segments[i-1].end
                if pause_before >= min_silence_duration:
                    pause_type = self._classify_pause_type(
                        result.segments[i-1],
                        segment,
                        pause_before
                    )
            
            analyzed_seg = SemanticSegment(
                segment=segment,
                segment_type=seg_type,
                semantic_importance=importance,
                emotion_markers=emotions,
                contains_filler=has_filler,
                filler_words=fillers,
                pause_before=pause_before,
                pause_type_before=pause_type
            )
            
            analyzed_segments.append(analyzed_seg)
        
        logger.info(f"Analyzed {len(analyzed_segments)} segments")
        return analyzed_segments

    def _classify_segment_type(self, text: str) -> str:
        """Classify segment type
        
        Args:
            text: Segment text
            
        Returns:
            One of: 'statement', 'question', 'emphasis', 'thought'
        """
        text_lower = text.lower().strip()
        
        # Check for question
        if any(marker in text for marker in self.QUESTION_MARKERS):
            return "question"
        
        # Check for emphasis
        if any(marker in text for marker in self.EMPHASIS_MARKERS):
            return "emphasis"
        
        # Check for thought/hesitation
        if any(marker in text for marker in self.THOUGHT_MARKERS):
            return "thought"
        
        return "statement"

    def _calculate_importance(self, text: str) -> float:
        """Calculate semantic importance score
        
        Args:
            text: Segment text
            
        Returns:
            Importance score 0.0-1.0
        """
        importance = 0.5  # Baseline
        
        # Increase for emphasis markers
        for marker in self.EMPHASIS_MARKERS:
            if marker in text:
                importance += 0.15
        
        # Increase for length (longer = potentially more important)
        word_count = len(text)
        if word_count > 20:
            importance += 0.1
        
        # Decrease for filler words
        if any(word in text for words in self.FILLER_WORDS.values() for word in words):
            importance -= 0.1
        
        return min(importance, 1.0)

    def _extract_emotion_markers(self, text: str) -> List[str]:
        """Extract emotion-related markers
        
        Args:
            text: Segment text
            
        Returns:
            List of emotion markers found
        """
        markers = []
        
        # Excitement/emphasis
        if any(word in text for word in ["太", "真的", "非常", "特别", "啊"]):
            markers.append("excitement")
        
        # Question/curiosity
        if any(word in text for word in self.QUESTION_MARKERS):
            markers.append("inquiry")
        
        # Uncertainty/thought
        if any(word in text for word in self.THOUGHT_MARKERS):
            markers.append("hesitation")
        
        # Negation
        if any(word in text for word in ["不", "没有", "不过"]):
            markers.append("negation")
        
        return markers

    def _extract_fillers(self, text: str) -> tuple[bool, List[str]]:
        """Extract filler words
        
        Args:
            text: Segment text
            
        Returns:
            Tuple of (has_filler, list_of_fillers)
        """
        found_fillers = []
        
        for category, words in self.FILLER_WORDS.items():
            for word in words:
                if word in text:
                    found_fillers.append(word)
        
        return len(found_fillers) > 0, found_fillers

    def _classify_pause_type(
        self,
        prev_segment: TranscriptSegment,
        current_segment: TranscriptSegment,
        pause_duration: float
    ) -> StopPauseType:
        """Classify type of pause between segments
        
        Args:
            prev_segment: Previous segment
            current_segment: Current segment
            pause_duration: Pause duration in seconds
            
        Returns:
            StopPauseType classification
        """
        prev_text = prev_segment.text.lower()
        curr_text = current_segment.text.lower()
        
        # Breath pause (short, natural)
        if pause_duration < 0.5:
            return StopPauseType.BREATH
        
        # Check for emphasis context
        if any(marker in prev_text for marker in self.EMPHASIS_MARKERS):
            return StopPauseType.EMPHASIS
        
        # Check for thought/hesitation context
        if any(marker in prev_text for marker in self.THOUGHT_MARKERS):
            return StopPauseType.THOUGHT
        
        # Check for emotional context
        if any(marker in [prev_text, curr_text] for marker in ["我", "感觉", "觉得"]):
            return StopPauseType.EMOTION
        
        # Narrative pause (sentence ending)
        if prev_text.endswith(("。", "！", "？")):
            return StopPauseType.NARRATIVE
        
        # Default to redundant if very long
        if pause_duration > 2.0:
            return StopPauseType.REDUNDANT
        
        return StopPauseType.NARRATIVE

    def get_analysis_summary(self, analyzed_segments: List[SemanticSegment]) -> Dict:
        """Get summary statistics of analysis
        
        Args:
            analyzed_segments: List of analyzed segments
            
        Returns:
            Summary dictionary
        """
        return {
            "total_segments": len(analyzed_segments),
            "segments_with_fillers": sum(1 for s in analyzed_segments if s.contains_filler),
            "redundant_pauses": sum(
                1 for s in analyzed_segments
                if s.pause_type_before == StopPauseType.REDUNDANT
            ),
            "average_importance": sum(s.semantic_importance for s in analyzed_segments) / len(analyzed_segments) if analyzed_segments else 0,
            "segment_types": {
                "statement": sum(1 for s in analyzed_segments if s.segment_type == "statement"),
                "question": sum(1 for s in analyzed_segments if s.segment_type == "question"),
                "emphasis": sum(1 for s in analyzed_segments if s.segment_type == "emphasis"),
                "thought": sum(1 for s in analyzed_segments if s.segment_type == "thought"),
            }
        }
