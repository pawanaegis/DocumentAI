import os
import logging
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from io import BytesIO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image

logger = logging.getLogger()

class QwenVLModel:
    """Class to handle Qwen2.5-VL-7B-Instruct model for visual document processing"""
    
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    LOCAL_MODEL_PATH = os.environ.get("QWEN_LOCAL_MODEL_PATH", "models/Qwen2.5-VL-7B-Instruct")
    
    def __init__(self, use_local_first: bool = True):
        """
        Initialize the Qwen VL model
        
        Args:
            use_local_first: Whether to try using a local model before downloading
        """
        self.use_local_first = use_local_first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model, tokenizer and processor, downloading if needed"""
        try:
            model_path = self._get_model_path()
            
            logger.info(f"Loading Qwen2.5-VL-7B-Instruct model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model with reduced precision for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load processor for image processing
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
            logger.info("Qwen2.5-VL-7B-Instruct model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Qwen2.5-VL-7B-Instruct model: {str(e)}")
            raise RuntimeError(f"Failed to load Qwen2.5-VL-7B-Instruct model: {str(e)}")
    
    def _get_model_path(self) -> str:
        """
        Get the model path, checking local path first if configured
        
        Returns:
            Path to the model (local or HuggingFace model ID)
        """
        if self.use_local_first:
            local_path = Path(self.LOCAL_MODEL_PATH)
            
            # Check if model files exist locally
            if local_path.exists() and any(local_path.iterdir()):
                logger.info(f"Using local model at {local_path}")
                return str(local_path)
            else:
                logger.info(f"Local model not found at {local_path}, downloading from HuggingFace")
                
                # Ensure the directory exists for future downloads
                os.makedirs(local_path, exist_ok=True)
                
                return self.MODEL_ID
        else:
            return self.MODEL_ID
    
    def process_document(self, image: Union[Image.Image, bytes], prompt: str) -> Dict[str, Any]:
        """
        Process a document image with the Qwen VL model
        
        Args:
            image: PIL Image or image bytes
            prompt: Text prompt asking about the document
            
        Returns:
            Dictionary with model response and metadata
        """
        try:
            # Convert bytes to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            
            # Prepare the prompt with system message
            messages = [
                {"role": "system", "content": "You are a document analysis assistant that can extract information from images of documents like invoices, receipts, and bills."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,dummy", "image": image}}
                ]}
            ]
            
            # Generate response
            response = self.model.chat(self.tokenizer, messages)
            
            return {
                "answer": response,
                "metadata": {
                    "model": self.MODEL_ID,
                    "device": self.device
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document with Qwen VL: {str(e)}")
            raise RuntimeError(f"Failed to process document with Qwen VL: {str(e)}")
    
    @staticmethod
    def is_model_available_locally() -> bool:
        """
        Check if the model is available locally
        
        Returns:
            True if model files exist locally, False otherwise
        """
        local_path = Path(QwenVLModel.LOCAL_MODEL_PATH)
        # Check if directory exists and contains model files
        return local_path.exists() and any(local_path.iterdir())
