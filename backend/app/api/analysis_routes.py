"""Content Analysis API Routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger
from pathlib import Path
import json

from app.core.database import get_db
from app.core.config import settings
from app.analysis.analyzer import ContentAnalyzer

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

_analyzer = None

def get_analyzer():
    """Get or initialize content analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ContentAnalyzer()
    return _analyzer

@router.post("/analyze/{upload_id}")
async def analyze_content(
    upload_id: str,
    db: Session = Depends(get_db),
):
    """
    Analyze content for issues (fillers, stutters, pauses, repetitions)
    
    Args:
        upload_id: Upload file ID
    
    Returns:
        Comprehensive analysis result
    """
    try:
        # Get parsed content
        parsing_dir = settings.CACHE_DIR / "parsing"
        parsing_path = parsing_dir / f"{upload_id}_parsing.json"
        
        if not parsing_path.exists():
            raise HTTPException(status_code=404, detail="Parsing data not found. Please parse content first.")
        
        with open(parsing_path, "r", encoding="utf-8") as f:
            parsing_data = json.load(f)
        
        analyzer = get_analyzer()
        
        logger.info(f"Starting content analysis for upload: {upload_id}")
        
        # Analyze content
        analysis = analyzer.analyze_content(parsing_data["sentences"])
        
        # Save analysis
        analysis_dir = settings.CACHE_DIR / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = analysis_dir / f"{upload_id}_analysis.json"
        
        analyzer.save_analysis(analysis, str(analysis_path))
        
        logger.info(f"Content analysis completed for {upload_id}")
        
        return {
            "upload_id": upload_id,
            "analysis_summary": analysis["summary"],
            "filler_words_count": analysis["filler_words"]["count"],
            "stutters_count": analysis["stutters"]["count"],
            "pauses_to_optimize": analysis["pauses"]["need_optimization"],
            "repetitions_count": analysis["repetitions"]["count"],
            "total_potential_edits": analysis["summary"]["total_potential_edits"],
            "analysis_file": str(analysis_path),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{upload_id}")
async def get_analysis_results(
    upload_id: str,
    db: Session = Depends(get_db),
):
    """
    Get full analysis results
    """
    try:
        analysis_dir = settings.CACHE_DIR / "analysis"
        analysis_path = analysis_dir / f"{upload_id}_analysis.json"
        
        if not analysis_path.exists():
            raise HTTPException(status_code=404, detail="Analysis results not found")
        
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        
        return analysis
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
