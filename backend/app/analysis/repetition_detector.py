"""Content Analysis Module - Repetition Detection"""
from typing import List, Dict, Any
from loguru import logger
from sentence_transformers import SentenceTransformer, util
import torch

class RepetitionDetector:
    """Detect repeated sentences and phrases using semantic similarity"""
    
    def __init__(self, model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"):
        """
        Initialize RepetitionDetector with semantic model
        
        Args:
            model_name: Sentence transformer model name
        """
        logger.info(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.95
        logger.info("RepetitionDetector initialized")
    
    def detect_repetitions(
        self,
        sentences: List[Dict[str, Any]],
        window_seconds: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Detect repeated sentences within time window
        
        Args:
            sentences: List of sentences with timing data
            window_seconds: Time window to check for repetitions
        
        Returns:
            List of repetition groups
        """
        try:
            logger.info(f"Detecting repetitions within {window_seconds}s window")
            
            if len(sentences) < 2:
                return []
            
            # Encode all sentences
            sentence_texts = [s["text"] for s in sentences]
            embeddings = self.model.encode(sentence_texts, convert_to_tensor=True)
            
            repetitions = []
            checked_pairs = set()
            
            # Compare sentences within time window
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    if (i, j) in checked_pairs:
                        continue
                    
                    sent_i = sentences[i]
                    sent_j = sentences[j]
                    
                    # Check if within time window
                    time_diff = sent_j["start"] - sent_i["start"]
                    if time_diff > window_seconds:
                        break
                    
                    # Calculate semantic similarity
                    similarity = float(util.pytorch_cos_sim(embeddings[i], embeddings[j])[0][0])
                    
                    # Only flag complete repetitions
                    if similarity >= self.similarity_threshold:
                        repetitions.append({
                            "first_sentence_id": sent_i["id"],
                            "second_sentence_id": sent_j["id"],
                            "first_text": sent_i["text"],
                            "second_text": sent_j["text"],
                            "first_start": sent_i["start"],
                            "first_end": sent_i["end"],
                            "second_start": sent_j["start"],
                            "second_end": sent_j["end"],
                            "time_between": round(time_diff, 3),
                            "similarity_score": round(similarity, 3),
                            "is_exact_duplicate": sentence_texts[i].strip() == sentence_texts[j].strip(),
                            "duration_first": round(sent_i["end"] - sent_i["start"], 3),
                            "duration_second": round(sent_j["end"] - sent_j["start"], 3),
                            "removable": True,
                        })
                        checked_pairs.add((i, j))
            
            logger.info(f"Detected {len(repetitions)} repetitions")
            return repetitions
        
        except Exception as e:
            logger.error(f"Repetition detection failed: {str(e)}")
            raise
