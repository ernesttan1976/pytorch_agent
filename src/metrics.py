"""Evaluation metrics for fine-tuned models."""
import numpy as np
from typing import List, Dict, Any, Optional
import torch


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
    
    Returns:
        Perplexity (exp(loss))
    """
    return np.exp(loss)


def compute_metrics(eval_pred, tokenizer=None) -> Dict[str, float]:
    """Compute metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object from transformers
        tokenizer: Optional tokenizer for decoding
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Flatten predictions and labels
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    # Mask out padding tokens (typically -100)
    mask = labels_flat != -100
    predictions_flat = predictions_flat[mask]
    labels_flat = labels_flat[mask]
    
    # Calculate perplexity if we have logits
    if len(predictions_flat) > 0 and predictions_flat.max() > 1:
        # Predictions are logits, need to get predicted tokens
        predicted_ids = np.argmax(predictions, axis=-1)
        predicted_flat = predicted_ids.flatten()[mask]
        
        # Calculate accuracy
        accuracy = (predicted_flat == labels_flat).mean()
        
        # For perplexity, we need the actual loss which should be computed separately
        return {"accuracy": float(accuracy)}
    else:
        # Predictions are already token IDs
        accuracy = (predictions_flat == labels_flat).mean()
        return {"accuracy": float(accuracy)}


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores (requires rouge-score package).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores["rouge1"].append(score["rouge1"].fmeasure)
            scores["rouge2"].append(score["rouge2"].fmeasure)
            scores["rougeL"].append(score["rougeL"].fmeasure)
        
        return {
            "rouge1": float(np.mean(scores["rouge1"])),
            "rouge2": float(np.mean(scores["rouge2"])),
            "rougeL": float(np.mean(scores["rougeL"]))
        }
    except ImportError:
        print("Warning: rouge-score package not installed. Install with: pip install rouge-score")
        return {}


def calculate_f1_score(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Calculate F1 score between token sequences.
    
    Args:
        pred_tokens: List of predicted tokens
        ref_tokens: List of reference tokens
    
    Returns:
        F1 score
    """
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    
    if len(pred_set) == 0 and len(ref_set) == 0:
        return 1.0
    
    if len(pred_set) == 0 or len(ref_set) == 0:
        return 0.0
    
    intersection = pred_set & ref_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(ref_set) if ref_set else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def exact_match(prediction: str, reference: str, normalize: bool = True) -> bool:
    """Check if prediction exactly matches reference.
    
    Args:
        prediction: Predicted text
        reference: Reference text
        normalize: Whether to normalize whitespace
    
    Returns:
        True if exact match
    """
    if normalize:
        prediction = " ".join(prediction.split())
        reference = " ".join(reference.split())
    
    return prediction.strip() == reference.strip()

