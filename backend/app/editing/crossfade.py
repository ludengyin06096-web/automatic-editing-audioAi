"""Editing Engine - Crossfade Implementation"""
from typing import Tuple
from loguru import logger
import numpy as np

class CrossfadeEngine:
    """Handle equal power crossfade between segments"""
    
    def __init__(self, crossfade_ms: int = 25, sample_rate: int = 16000):
        """
        Initialize CrossfadeEngine
        
        Args:
            crossfade_ms: Crossfade duration in milliseconds
            sample_rate: Sample rate
        """
        self.crossfade_ms = crossfade_ms
        self.sample_rate = sample_rate
        self.crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        logger.info(f"CrossfadeEngine initialized (Duration: {crossfade_ms}ms, Samples: {self.crossfade_samples})")
    
    def equal_power_crossfade(
        self,
        segment1: np.ndarray,
        segment2: np.ndarray,
    ) -> np.ndarray:
        """
        Perform equal power crossfade between two segments
        
        Uses equal power curve to maintain constant loudness:
        - Fade out: sqrt(1 - t^2)
        - Fade in: sqrt(t^2)
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
        
        Returns:
            Crossfaded audio segments
        """
        try:
            # Ensure minimum segment length
            if len(segment1) < self.crossfade_samples or len(segment2) < self.crossfade_samples:
                logger.warning("Segment too short for crossfade, using linear crossfade")
                return self._linear_crossfade(segment1, segment2)
            
            # Extract fade regions
            fade_out = segment1[-self.crossfade_samples:].copy()
            fade_in = segment2[:self.crossfade_samples].copy()
            
            # Create equal power curves
            t = np.linspace(0, 1, self.crossfade_samples)
            fade_out_curve = np.sqrt(1 - t**2)
            fade_in_curve = np.sqrt(t**2)
            
            # Apply crossfade
            crossfaded = fade_out * fade_out_curve + fade_in * fade_in_curve
            
            # Combine segments
            # Remove fade regions from original segments
            result = np.concatenate([
                segment1[:-self.crossfade_samples],  # First segment without fade-out
                crossfaded,                           # Crossfaded section
                segment2[self.crossfade_samples:],   # Second segment without fade-in
            ])
            
            logger.info(f"Crossfade applied: {self.crossfade_ms}ms")
            return result
        
        except Exception as e:
            logger.error(f"Crossfade failed: {str(e)}")
            raise
    
    def _linear_crossfade(
        self,
        segment1: np.ndarray,
        segment2: np.ndarray,
    ) -> np.ndarray:
        """
        Perform simple linear crossfade (fallback)
        """
        try:
            # Use minimum segment length as fade length
            fade_len = min(len(segment1), len(segment2), self.crossfade_samples)
            
            t = np.linspace(0, 1, fade_len)
            fade_out = segment1[-fade_len:] * (1 - t)
            fade_in = segment2[:fade_len] * t
            crossfaded = fade_out + fade_in
            
            result = np.concatenate([
                segment1[:-fade_len],
                crossfaded,
                segment2[fade_len:],
            ])
            
            logger.info(f"Linear crossfade applied: {fade_len} samples")
            return result
        
        except Exception as e:
            logger.error(f"Linear crossfade failed: {str(e)}")
            raise
    
    def validate_crossfade_point(
        self,
        segment1: np.ndarray,
        segment2: np.ndarray,
    ) -> bool:
        """
        Validate if crossfade point is suitable
        
        Args:
            segment1: First segment
            segment2: Second segment
        
        Returns:
            True if crossfade is suitable
        """
        try:
            # Check energy levels at crossfade point
            energy1 = np.sqrt(np.mean(segment1[-self.crossfade_samples:]**2))
            energy2 = np.sqrt(np.mean(segment2[:self.crossfade_samples]**2))
            
            # Energy difference should not exceed threshold (avoid harsh crossfade)
            energy_ratio = max(energy1, energy2) / (min(energy1, energy2) + 1e-10)
            
            is_valid = energy_ratio < 3.0  # 3x difference is acceptable
            
            logger.info(f"Crossfade point validation: energy_ratio={energy_ratio:.2f}, valid={is_valid}")
            return is_valid
        
        except Exception as e:
            logger.error(f"Crossfade validation failed: {str(e)}")
            return False
