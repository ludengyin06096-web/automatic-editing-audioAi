"""Audio loading and conversion module"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
from pydub import AudioSegment


class AudioLoader:
    """Load and normalize audio files in various formats"""

    SUPPORTED_FORMATS = (".wav", ".mp3", ".m4a", ".flac")

    def __init__(self, target_sample_rate: int = 16000):
        """Initialize audio loader
        
        Args:
            target_sample_rate: Target sample rate for all loaded audio
        """
        self.target_sample_rate = target_sample_rate

    def load(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        try:
            # Use librosa for consistent loading
            waveform, sr = librosa.load(
                str(file_path),
                sr=self.target_sample_rate,
                mono=True
            )
            logger.info(
                f"Loaded audio: {file_path.name} | "
                f"Sample Rate: {sr}Hz | Duration: {len(waveform)/sr:.2f}s"
            )
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    def load_stereo(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file as stereo
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate) where waveform is (channels, samples)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            waveform, sr = librosa.load(
                str(file_path),
                sr=self.target_sample_rate,
                mono=False
            )
            logger.info(
                f"Loaded stereo audio: {file_path.name} | "
                f"Channels: {waveform.shape[0]} | Sample Rate: {sr}Hz"
            )
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading stereo audio {file_path}: {e}")
            raise

    def save(
        self,
        waveform: np.ndarray,
        output_path: Path,
        sample_rate: int,
        bit_depth: int = 16
    ) -> None:
        """Save audio to file
        
        Args:
            waveform: Audio waveform
            output_path: Output file path
            sample_rate: Sample rate
            bit_depth: Bit depth (16 or 24)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure waveform is in valid range
        waveform = np.clip(waveform, -1.0, 1.0)
        
        try:
            sf.write(
                str(output_path),
                waveform,
                sample_rate,
                subtype=f"PCM_{bit_depth}"
            )
            logger.info(f"Saved audio: {output_path.name}")
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            raise

    def convert_format(
        self,
        input_path: Path,
        output_path: Path,
        output_format: str = "wav",
        bitrate: str = "128k"
    ) -> None:
        """Convert audio between formats
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            output_format: Output format (wav, mp3, m4a, flac)
            bitrate: Bitrate for lossy formats
        """
        try:
            audio = AudioSegment.from_file(str(input_path))
            audio.export(
                str(output_path),
                format=output_format.lstrip("."),
                bitrate=bitrate
            )
            logger.info(
                f"Converted {input_path.name} to "
                f"{output_format} ({bitrate})"
            )
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise

    def get_duration(self, file_path: Path) -> float:
        """Get audio duration in seconds
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(filename=str(file_path))
            return duration
        except Exception as e:
            logger.error(f"Error getting duration of {file_path}: {e}")
            raise

    def normalize(self, waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target loudness (dB)
        
        Args:
            waveform: Audio waveform
            target_db: Target loudness in dB
            
        Returns:
            Normalized waveform
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(waveform ** 2))
        
        if rms == 0:
            return waveform
        
        # Convert target dB to linear scale
        target_linear = 10 ** (target_db / 20.0)
        
        # Calculate scaling factor
        scale_factor = target_linear / rms
        
        # Apply scaling and clip
        normalized = np.clip(waveform * scale_factor, -1.0, 1.0)
        
        return normalized
