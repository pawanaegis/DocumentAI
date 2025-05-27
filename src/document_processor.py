import io
import os
import tempfile
from typing import Dict, Any, Tuple, List, Optional, Union
import base64
import logging

import pytesseract
from PIL import Image
import pdf2image
import PyPDF2
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests

from exceptions import DocumentProcessingError, LLMProcessingError
from qwen_vl_model import QwenVLModel

logger = logging.getLogger()

class DocumentProcessor:
    """Class to process documents and extract information based on prompts"""
    
    def __init__(self, use_local_model_first: bool = True):
        """Initialize the document processor"""
        # Qwen model configuration
        self.use_local_model_first = use_local_model_first
        self.qwen_model = None
        
        # Initialize Qwen model
        try:
            logger.info("Initializing Qwen2.5-VL-7B-Instruct model")
            self.qwen_model = QwenVLModel(use_local_first=use_local_model_first)
            logger.info("Qwen2.5-VL-7B-Instruct model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2.5-VL-7B-Instruct model: {str(e)}")
            raise RuntimeError(f"Failed to initialize Qwen2.5-VL-7B-Instruct model: {str(e)}")
        
        # Define prompt template for document question answering
        self.prompt_template = PromptTemplate(
            input_variables=["document_text", "question"],
            template="""
            You are an AI assistant specialized in extracting information from documents like invoices, receipts, and bills.
            
            Below is the text extracted from a document:
            
            {document_text}
            
            Based on the document text above, please answer the following question:
            {question}
            
            If the information is not present in the document, respond with "Information not found in document."
            Be precise and only extract the specific information requested.
            """
        )
    
    def process_document(self, document_bytes: bytes, document_type: str, prompt: str) -> Dict[str, Any]:
        """
        Process the document and extract information based on the prompt
        
        Args:
            document_bytes: Binary content of the document
            document_type: MIME type of the document (image/jpeg, image/png, application/pdf)
            prompt: User's question about the document
            
        Returns:
            Dictionary with extracted information and metadata
        """
        if document_type.startswith('image/'):
            # Direct processing for images using Qwen VL model
            logger.info("Using Qwen2.5-VL-7B-Instruct model for visual document processing")
            return self._process_with_qwen(document_bytes, prompt)
        
        elif document_type == 'application/pdf':
            # For PDFs, extract text and then use Qwen for text processing
            document_text = self._extract_text(document_bytes, document_type)
            
            # Convert text to prompt for Qwen
            text_prompt = f"The following is text extracted from a PDF document. Based on this text, {prompt}\n\nDocument text: {document_text}"
            
            # Process with Qwen using text-only approach
            result = self._process_with_qwen_text(text_prompt)
            
            return {
                "answer": result,
                "confidence": self._calculate_confidence(result),
                "metadata": {
                    "document_type": document_type,
                    "text_length": len(document_text),
                    "model": "Qwen2.5-VL-7B-Instruct",
                    "processing_method": "pdf_text_extraction_qwen"
                }
            }
        
        else:
            # For other document types
            raise DocumentProcessingError(f"Unsupported document type: {document_type}")
    
    def _extract_text(self, document_bytes: bytes, document_type: str) -> str:
        """
        Extract text from document bytes based on document type
        
        Args:
            document_bytes: Binary content of the document
            document_type: MIME type of the document
            
        Returns:
            Extracted text from the document
        """
        try:
            if document_type.startswith('image/'):
                return self._extract_text_from_image(document_bytes)
            elif document_type == 'application/pdf':
                return self._extract_text_from_pdf(document_bytes)
            else:
                raise DocumentProcessingError(f"Unsupported document type: {document_type}")
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise DocumentProcessingError(f"Failed to extract text from document: {str(e)}")
    
    def _extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(image)
            
            return text
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise DocumentProcessingError(f"OCR processing failed: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF document"""
        try:
            # First try to extract text directly from PDF
            pdf_text = self._extract_text_from_pdf_direct(pdf_bytes)
            
            # If direct extraction yields little text, try OCR
            if len(pdf_text.strip()) < 100:
                logger.info("Direct PDF text extraction yielded little text, trying OCR")
                pdf_text = self._extract_text_from_pdf_ocr(pdf_bytes)
                
            return pdf_text
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise DocumentProcessingError(f"PDF processing failed: {str(e)}")
    
    def _extract_text_from_pdf_direct(self, pdf_bytes: bytes) -> str:
        """Extract text directly from PDF using PyPDF2"""
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
            
        return text
    
    def _extract_text_from_pdf_ocr(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using OCR"""
        # Convert PDF to images
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(temp_pdf_path)
            
            # Extract text from each image
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
                
            return text
        finally:
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    # Mistral API method has been removed as we're now using Qwen exclusively
    
    def _process_with_qwen(self, document_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """
        Process document using the Qwen2.5-VL-7B-Instruct model
        
        Args:
            document_bytes: Binary content of the document
            prompt: User's question about the document
            
        Returns:
            Dictionary with extracted information and metadata
        """
        if not self.qwen_model:
            raise RuntimeError("Qwen model not initialized")
        
        # Process the document with Qwen VL model
        result = self.qwen_model.process_document(document_bytes, prompt)
        
        # Add confidence score to the result
        result["confidence"] = self._calculate_confidence(result["answer"])
        
        # Add processing method to metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["processing_method"] = "qwen_vl_direct"
        
        return result
        
    def _process_with_qwen_text(self, text_prompt: str) -> str:
        """
        Process text-only content using the Qwen2.5-VL-7B-Instruct model
        
        Args:
            text_prompt: Text prompt to send to the model
            
        Returns:
            Model response as a string
        """
        if not self.qwen_model:
            raise RuntimeError("Qwen model not initialized")
            
        # Create a text-only message for the model
        messages = [
            {"role": "system", "content": "You are a document analysis assistant that extracts information from text documents."},
            {"role": "user", "content": text_prompt}
        ]
        
        # Generate response using the chat method
        response = self.qwen_model.model.chat(self.qwen_model.tokenizer, messages)
        
        return response
    
    def _calculate_confidence(self, response: str) -> float:
        """
        Calculate a confidence score for the response
        
        This is a simple implementation. In a production system, you might 
        want to use more sophisticated methods.
        """
        if "Information not found in document" in response:
            return 0.0
        
        # Simple heuristic based on response length
        # More sophisticated approaches could be implemented
        confidence = min(0.9, max(0.5, len(response) / 200))
        
        return confidence
