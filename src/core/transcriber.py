"""Speech-to-text transcription module using Whisper and WhisperX"""

import numpy as np
import whisper
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
from src.config import settings


@dataclass
class TranscriptSegment:
    """Single transcription segment with timing"""
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None


@dataclass
class TranscriptResult:
    """Complete transcription result"""
    text: str
    language: str
    segments: List[TranscriptSegment]
    duration: float


class Transcriber:
    """Whisper-based transcriber with WhisperX for word-level timestamps"""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "zh"
    ):
        """Initialize transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (float32, float16, int8)
            language: Language code (zh for Chinese)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(
            model_name,
            device=device
        )
        logger.info(f"Whisper model loaded successfully")
        
        # Try to load WhisperX for word-level timestamps
        try:
            import whisperx
            self.whisperx_available = True
            logger.info("WhisperX available for word-level timestamps")
        except ImportError:
            self.whisperx_available = False
            logger.warning("WhisperX not available, using segment-level timestamps only")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> TranscriptResult:
        """Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (defaults to instance language)
            
        Returns:
            TranscriptResult with segments and full text
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        lang = language or self.language
        
        try:
            logger.info(f"Transcribing: {audio_path.name}")
            
            result = self.model.transcribe(
                str(audio_path),
                language=lang,
                task="transcribe",
                verbose=False,
                temperature=0.0,  # Deterministic
                fp16=self.compute_type == "float16"
            )
            
            # Parse segments
            segments = [
                TranscriptSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("confidence", 0.95)
                )
                for seg in result["segments"]
            ]
            
            # Get full text
            full_text = " ".join([seg.text for seg in segments])
            
            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"Duration: {result['duration']:.2f}s"
            )
            
            return TranscriptResult(
                text=full_text,
                language=result.get("language", lang),
                segments=segments,
                duration=result["duration"]
            )
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            raise

    def transcribe_with_word_timestamps(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Dict:
        """Transcribe with word-level timestamps using WhisperX
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with word-level timing information
        """
        if not self.whisperx_available:
            logger.warning("WhisperX not available, falling back to segment-level")
            result = self.transcribe(audio_path, language)
            return asdict(result)
        
        try:
            import whisperx
            
            lang = language or self.language
            device = self.device
            batch_size = 8 if device == "cuda" else 1
            
            logger.info(f"Transcribing with WhisperX: {audio_path.name}")
            
            # Load audio
            audio = whisperx.load_audio(str(audio_path))
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language=lang,
                batch_size=batch_size
            )
            
            # Align words
            model_a, metadata = whisperx.load_align_model(
                language_code=lang,
                device=device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                interpolate_method="nearest"
            )
            
            logger.info("WhisperX alignment complete")
            return result
            
        except Exception as e:
            logger.error(f"Error with WhisperX transcription: {e}")
            raise

    def save_transcript(
        self,
        result: TranscriptResult,
        output_path: Path,
        format: str = "json"
    ) -> None:
        """Save transcription result to file
        
        Args:
            result: TranscriptResult object
            output_path: Output file path
            format: Output format (json, txt, srt, vtt)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(result), f, ensure_ascii=False, indent=2)
            
            elif format == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.text)
            
            elif format == "srt":
                self._save_as_srt(result, output_path)
            
            elif format == "vtt":
                self._save_as_vtt(result, output_path)
            
            logger.info(f"Saved transcript: {output_path.name}")
        
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise

    @staticmethod
    def _save_as_srt(result: TranscriptResult, output_path: Path) -> None:
        """Save as SRT subtitle format"""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result.segments, 1):
                start = Transcriber._format_timestamp(seg.start)
                end = Transcriber._format_timestamp(seg.end)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg.text}\n\n")

    @staticmethod
    def _save_as_vtt(result: TranscriptResult, output_path: Path) -> None:
        """Save as VTT subtitle format"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in result.segments:
                start = Transcriber._format_timestamp(seg.start)
                end = Transcriber._format_timestamp(seg.end)
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg.text}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float, milliseconds: bool = True) -> str:
        """Format timestamp for subtitle formats
        
        Args:
            seconds: Time in seconds
            milliseconds: Include milliseconds (for SRT/VTT)
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        if milliseconds:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
