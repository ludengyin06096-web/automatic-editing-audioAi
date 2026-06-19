"""Editing Engine API Routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger
from pathlib import Path
import json

from app.core.database import get_db
from app.core.config import settings
from app.editing.editor import EditingEngine
from app.editing.segmenter import AudioSegmenter

router = APIRouter(prefix="/api/editing", tags=["editing"])

_editor = None
_segmenter = None

def get_editor():
    """Get or initialize editing engine"""
    global _editor
    if _editor is None:
        _editor = EditingEngine(
            sample_rate=settings.SAMPLE_RATE,
            crossfade_ms=settings.CROSSFADE_MS,
        )
    return _editor

def get_segmenter():
    """Get or initialize segmenter"""
    global _segmenter
    if _segmenter is None:
        _segmenter = AudioSegmenter(sample_rate=settings.SAMPLE_RATE)
    return _segmenter

@router.post("/plan/{upload_id}")
async def plan_edits(
    upload_id: str,
    db: Session = Depends(get_db),
):
    """
    Plan edits based on analysis
    
    Args:
        upload_id: Upload file ID
    
    Returns:
        Edit plan
    """
    try:
        # Get analysis result
        analysis_dir = settings.CACHE_DIR / "analysis"
        analysis_path = analysis_dir / f"{upload_id}_analysis.json"
        
        if not analysis_path.exists():
            raise HTTPException(status_code=404, detail="Analysis not found. Please analyze content first.")
        
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        
        editor = get_editor()
        
        logger.info(f"Planning edits for upload: {upload_id}")
        
        # Plan edits
        edits = editor.plan_edits(analysis)
        
        # Save edit plan
        editing_dir = settings.CACHE_DIR / "editing"
        editing_dir.mkdir(parents=True, exist_ok=True)
        editing_plan_path = editing_dir / f"{upload_id}_edit_plan.json"
        
        with open(editing_plan_path, "w", encoding="utf-8") as f:
            json.dump(edits, f, ensure_ascii=False, indent=2)
        
        # Generate edit log
        edit_log = editor.generate_edit_log(edits)
        edit_log_path = editing_dir / f"{upload_id}_edit_log.json"
        
        with open(edit_log_path, "w", encoding="utf-8") as f:
            json.dump(edit_log, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Edit plan completed for {upload_id}: {len(edits)} edits")
        
        return {
            "upload_id": upload_id,
            "total_edits_planned": len(edits),
            "edits_by_type": {
                "remove_filler": len([e for e in edits if e["type"] == "remove_filler"]),
                "remove_stutter": len([e for e in edits if e["type"] == "remove_stutter"]),
                "compress_pause": len([e for e in edits if e["type"] == "compress_pause"]),
                "remove_repetition": len([e for e in edits if e["type"] == "remove_repetition"]),
            },
            "edit_plan_file": str(editing_plan_path),
            "edit_log_file": str(edit_log_path),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edit planning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plan/{upload_id}")
async def get_edit_plan(
    upload_id: str,
    db: Session = Depends(get_db),
):
    """
    Get edit plan details
    """
    try:
        editing_dir = settings.CACHE_DIR / "editing"
        editing_plan_path = editing_dir / f"{upload_id}_edit_plan.json"
        
        if not editing_plan_path.exists():
            raise HTTPException(status_code=404, detail="Edit plan not found")
        
        with open(editing_plan_path, "r", encoding="utf-8") as f:
            edits = json.load(f)
        
        return {
            "upload_id": upload_id,
            "total_edits": len(edits),
            "edits": edits,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get edit plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
