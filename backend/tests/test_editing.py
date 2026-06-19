"""Tests for Editing Engine"""
import pytest
import numpy as np
from app.editing.crossfade import CrossfadeEngine
from app.editing.segmenter import AudioSegmenter

def test_crossfade_engine_initialization():
    """Test crossfade engine initialization"""
    engine = CrossfadeEngine(crossfade_ms=25, sample_rate=16000)
    assert engine.crossfade_ms == 25
    assert engine.sample_rate == 16000
    assert engine.crossfade_samples == 400  # 25ms at 16kHz

def test_equal_power_crossfade():
    """Test equal power crossfade"""
    engine = CrossfadeEngine(crossfade_ms=10, sample_rate=16000)
    
    # Create test segments
    seg1 = np.sin(np.linspace(0, np.pi, 16000))  # 1 second sine wave
    seg2 = np.cos(np.linspace(0, np.pi, 16000))  # 1 second cosine wave
    
    result = engine.equal_power_crossfade(seg1, seg2)
    
    # Result should be shorter than combined (due to overlap)
    assert len(result) < len(seg1) + len(seg2)
    # Result should not have NaN or Inf values
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

def test_audio_segmenter_extract_segment():
    """Test audio segment extraction"""
    segmenter = AudioSegmenter(sample_rate=16000)
    
    # Create test audio (1 second at 16kHz)
    audio = np.sin(np.linspace(0, 2*np.pi, 16000))
    
    # Extract 0.2 to 0.5 seconds
    segment = segmenter.extract_segment(audio, 0.2, 0.5, 16000)
    
    # Should be 0.3 seconds long
    expected_length = int(0.3 * 16000)
    assert len(segment) == expected_length

def test_crossfade_validation():
    """Test crossfade point validation"""
    engine = CrossfadeEngine(crossfade_ms=25, sample_rate=16000)
    
    seg1 = np.random.randn(16000)
    seg2 = np.random.randn(16000)
    
    is_valid = engine.validate_crossfade_point(seg1, seg2)
    
    # Should return boolean
    assert isinstance(is_valid, (bool, np.bool_))
