"""Tests for Content Analysis"""
import pytest
from app.analysis.filler_detector import FillerWordDetector
from app.analysis.stutter_detector import StutterDetector
from app.analysis.pause_optimizer import PauseOptimizer

# Sample test data
SAMPLE_SENTENCES = [
    {
        "id": 0,
        "text": "那个这是一个测试",
        "start": 0.0,
        "end": 2.5,
        "words": [
            {"id": 0, "word": "那个", "start": 0.0, "end": 0.3},
            {"id": 1, "word": "这", "start": 0.4, "end": 0.6},
            {"id": 2, "word": "是", "start": 0.7, "end": 0.9},
            {"id": 3, "word": "一个", "start": 1.0, "end": 1.3},
            {"id": 4, "word": "测试", "start": 1.4, "end": 2.5},
        ]
    },
    {
        "id": 1,
        "text": "我我我觉得很好",
        "start": 3.0,
        "end": 5.5,
        "words": [
            {"id": 5, "word": "我", "start": 3.0, "end": 3.2},
            {"id": 6, "word": "我", "start": 3.3, "end": 3.5},
            {"id": 7, "word": "我", "start": 3.6, "end": 3.8},
            {"id": 8, "word": "觉得", "start": 4.0, "end": 4.5},
            {"id": 9, "word": "很好", "start": 5.0, "end": 5.5},
        ]
    },
]

def test_filler_word_detector():
    """Test filler word detection"""
    detector = FillerWordDetector()
    
    results = detector.detect_filler_words(SAMPLE_SENTENCES)
    
    # Should find "那个" as filler
    assert len(results) > 0
    assert any(r["word"] == "那个" for r in results)

def test_stutter_detector():
    """Test stutter detection"""
    detector = StutterDetector()
    
    results = detector.detect_stutters(SAMPLE_SENTENCES)
    
    # Should find "我我我" as stutter
    assert len(results) > 0
    assert any(r["word"] == "我" for r in results)

def test_pause_optimizer():
    """Test pause detection"""
    optimizer = PauseOptimizer(pause_threshold=0.5)
    
    # Modify sample to have a pause
    test_sentences = [{
        "id": 0,
        "text": "测试",
        "start": 0.0,
        "end": 2.0,
        "words": [
            {"id": 0, "word": "测", "start": 0.0, "end": 0.3},
            {"id": 1, "word": "试", "start": 1.5, "end": 2.0},
        ]
    }]
    
    results = optimizer.detect_pauses(test_sentences)
    
    # Should detect pause between "测" and "试"
    assert len(results) > 0
