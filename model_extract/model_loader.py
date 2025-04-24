"""
Model loading functions for Hugging Face Model Architecture Extractor
"""

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, AutoModelForMaskedLM,
    AutoModelForImageClassification, AutoModelForObjectDetection,
    AutoModelForImageSegmentation
)
from .config import logger

def load_model_for_task(model_id, task):
    """Load appropriate model class based on task type"""
    try:
        logger.info(f"Loading model {model_id} for task {task}")
        
        # Try task-specific model loader first
        try:
            if task == "image-classification":
                return AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
            elif task == "object-detection":
                return AutoModelForObjectDetection.from_pretrained(model_id, trust_remote_code=True)
            elif task == "image-segmentation":
                return AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
            elif task == "text-classification":
                return AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
            elif task == "text-generation":
                return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            elif task in ["translation", "summarization"]:
                return AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
            elif task == "question-answering":
                return AutoModelForQuestionAnswering.from_pretrained(model_id, trust_remote_code=True)
            elif task == "fill-mask":
                return AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
            elif task == "token-classification":
                return AutoModelForTokenClassification.from_pretrained(model_id, trust_remote_code=True)
            else:
                # Default to general model
                return AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e1:
            logger.warning(f"Task-specific model load failed: {e1}")
            
            # Try generic model loader
            try:
                logger.info(f"Trying generic AutoModel")
                return AutoModel.from_pretrained(model_id, trust_remote_code=True)
            except Exception as e2:
                logger.error(f"Generic model load failed: {e2}")
                return None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None 