"""Editing Engine - Audio Segmentation"""
from typing import List, Dict, Any, Tuple
from loguru import logger
import librosa
import numpy as np
from pydub import AudioSegment
from pathlib import Path

class AudioSegmenter:
    """Handle audio segmentation and cutting"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize AudioSegmenter
        
        Args:
            sample_rate: Target sample rate
        """
        self.sample_rate = sample_rate
        logger.info(f"AudioSegmenter initialized (SR: {sample_rate}Hz)")
    
    def load_audio(
        self,
        file_path: str,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            logger.info(f"Loading audio: {file_path}")
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            logger.info(f"Audio loaded: {len(y)} samples at {sr}Hz")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise
    
    def extract_segment(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract audio segment by time
        
        Args:
            audio: Audio array
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Sample rate
        
        Returns:
            Segment audio array
        """
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            segment = audio[start_sample:end_sample]
            logger.info(f"Extracted segment: {start_time}s-{end_time}s ({len(segment)} samples)")
            
            return segment
        except Exception as e:
            logger.error(f"Failed to extract segment: {str(e)}")
            raise
    
    def get_segment_by_word_ids(
        self,
        audio: np.ndarray,
        words: List[Dict[str, Any]],
        start_word_id: int,
        end_word_id: int,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract segment by word IDs
        
        Args:
            audio: Audio array
            words: Word timeline
            start_word_id: Start word ID
            end_word_id: End word ID
            sample_rate: Sample rate
        
        Returns:
            Segment audio array
        """
        try:
            # Find word boundaries
            start_word = None
            end_word = None
            
            for word in words:
                if word["id"] == start_word_id:
                    start_word = word
                if word["id"] == end_word_id:
                    end_word = word
            
            if not start_word or not end_word:
                raise ValueError(f"Word IDs not found: {start_word_id}-{end_word_id}")
            
            return self.extract_segment(
                audio,
                start_word["start"],
                end_word["end"],
                sample_rate,
            )
        except Exception as e:
            logger.error(f"Failed to get segment by word IDs: {str(e)}")
            raise
