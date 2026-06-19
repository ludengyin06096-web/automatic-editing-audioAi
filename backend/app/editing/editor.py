"""Editing Engine - Main Editor"""
from typing import List, Dict, Any
from loguru import logger
from pathlib import Path
import numpy as np
from datetime import datetime
import json

from app.editing.segmenter import AudioSegmenter
from app.editing.crossfade import CrossfadeEngine
from app.core.config import settings

class EditingEngine:
    """Main audio editing orchestrator"""
    
    def __init__(self, sample_rate: int = 16000, crossfade_ms: int = 25):
        """
        Initialize EditingEngine
        
        Args:
            sample_rate: Sample rate
            crossfade_ms: Crossfade duration in ms
        """
        self.sample_rate = sample_rate
        self.crossfade_ms = crossfade_ms
        self.segmenter = AudioSegmenter(sample_rate)
        self.crossfade_engine = CrossfadeEngine(crossfade_ms, sample_rate)
        
        self.edit_log = []
        logger.info("EditingEngine initialized")
    
    def plan_edits(
        self,
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Plan edits based on analysis
        
        Args:
            analysis: Content analysis result
        
        Returns:
            List of planned edits
        """
        try:
            edits = []
            edit_id = 0
            
            # Plan filler word removal
            for filler in analysis["filler_words"]["items"]:
                edits.append({
                    "id": edit_id,
                    "type": "remove_filler",
                    "start": filler["start"],
                    "end": filler["end"],
                    "word": filler["word"],
                    "priority": "high",
                    "applied": False,
                })
                edit_id += 1
            
            # Plan stutter removal
            for stutter in analysis["stutters"]["items"]:
                edits.append({
                    "id": edit_id,
                    "type": "remove_stutter",
                    "start": stutter["start"],
                    "end": stutter["end"],
                    "pattern": stutter["pattern"],
                    "repeat_count": stutter["repeat_count"],
                    "priority": "high",
                    "applied": False,
                })
                edit_id += 1
            
            # Plan pause compression
            for pause_opt in analysis["pauses"]["optimizations"]:
                edits.append({
                    "id": edit_id,
                    "type": "compress_pause",
                    "start": pause_opt["pause_id"],
                    "duration_current": pause_opt["current_duration"],
                    "duration_target": pause_opt["target_duration"],
                    "priority": "medium",
                    "applied": False,
                })
                edit_id += 1
            
            # Plan repetition removal
            for rep in analysis["repetitions"]["items"]:
                edits.append({
                    "id": edit_id,
                    "type": "remove_repetition",
                    "first_sentence_id": rep["first_sentence_id"],
                    "second_sentence_id": rep["second_sentence_id"],
                    "start": rep["second_start"],
                    "end": rep["second_end"],
                    "priority": "medium",
                    "applied": False,
                })
                edit_id += 1
            
            # Sort by start time
            edits.sort(key=lambda x: x["start"])
            
            logger.info(f"Planned {len(edits)} edits")
            return edits
        
        except Exception as e:
            logger.error(f"Edit planning failed: {str(e)}")
            raise
    
    def apply_edits(
        self,
        audio: np.ndarray,
        edits: List[Dict[str, Any]],
        words: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Apply planned edits to audio
        
        Args:
            audio: Original audio array
            edits: List of planned edits
            words: Word timeline
        
        Returns:
            Edited audio array
        """
        try:
            result_audio = audio.copy()
            applied_edits = []
            
            # Sort edits by start time (reverse) to avoid index shifting
            sorted_edits = sorted(edits, key=lambda x: x["start"], reverse=True)
            
            for edit in sorted_edits:
                try:
                    if edit["type"] == "remove_filler":
                        result_audio = self._remove_segment(result_audio, edit["start"], edit["end"])
                        applied_edits.append(edit)
                        
                    elif edit["type"] == "remove_stutter":
                        result_audio = self._remove_segment(result_audio, edit["start"], edit["end"])
                        applied_edits.append(edit)
                        
                    elif edit["type"] == "remove_repetition":
                        result_audio = self._remove_segment(result_audio, edit["start"], edit["end"])
                        applied_edits.append(edit)
                    
                    logger.info(f"Applied edit: {edit['type']}")
                
                except Exception as e:
                    logger.warning(f"Failed to apply edit {edit['id']}: {str(e)}")
                    # Fallback: continue with next edit
                    continue
            
            logger.info(f"Applied {len(applied_edits)} edits successfully")
            return result_audio
        
        except Exception as e:
            logger.error(f"Edit application failed: {str(e)}")
            raise
    
    def _remove_segment(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """
        Remove audio segment between start and end time
        
        Args:
            audio: Audio array
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Audio with segment removed
        """
        try:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            # Remove segment with crossfade at boundaries
            if start_sample > 0 and end_sample < len(audio):
                # Has both before and after segments
                before = audio[:start_sample]
                after = audio[end_sample:]
                
                # Apply crossfade if possible
                if len(before) > 0 and len(after) > 0:
                    result = self.crossfade_engine.equal_power_crossfade(before, after)
                else:
                    result = np.concatenate([before, after])
            else:
                # Only remove from beginning or end
                result = np.concatenate([
                    audio[:start_sample],
                    audio[end_sample:],
                ])
            
            logger.info(f"Removed segment: {start_time}s-{end_time}s")
            return result
        
        except Exception as e:
            logger.error(f"Segment removal failed: {str(e)}")
            raise
    
    def generate_edit_log(
        self,
        edits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate edit log report
        
        Args:
            edits: List of edits
        
        Returns:
            Edit log dictionary
        """
        try:
            log = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_edits_planned": len(edits),
                "edits_by_type": {},
                "edit_details": edits,
                "statistics": {
                    "remove_filler": len([e for e in edits if e["type"] == "remove_filler"]),
                    "remove_stutter": len([e for e in edits if e["type"] == "remove_stutter"]),
                    "compress_pause": len([e for e in edits if e["type"] == "compress_pause"]),
                    "remove_repetition": len([e for e in edits if e["type"] == "remove_repetition"]),
                },
            }
            
            logger.info(f"Edit log generated with {len(edits)} entries")
            return log
        
        except Exception as e:
            logger.error(f"Failed to generate edit log: {str(e)}")
            raise
