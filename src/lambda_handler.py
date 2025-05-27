import json
import base64
import os
import logging
from document_processor import DocumentProcessor
from exceptions import DocumentProcessingError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda handler for document AI extraction
    
    Expected input:
    {
        "document": "base64_encoded_document",
        "document_type": "image/jpeg|image/png|application/pdf",
        "prompt": "What is the invoice number?"
    }
    """
    try:
        # Parse input
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        
        # Get document and prompt
        document_base64 = body.get('document')
        document_type = body.get('document_type', 'application/pdf')
        prompt = body.get('prompt')
        
        # Validate input
        if not document_base64:
            return format_response(400, {"error": "Missing document"})
        
        if not prompt:
            return format_response(400, {"error": "Missing prompt"})
        
        # Process document
        processor = DocumentProcessor()
        
        # Decode document
        document_bytes = base64.b64decode(document_base64)
        
        # Process document and extract information based on prompt
        result = processor.process_document(document_bytes, document_type, prompt)
        
        # Return result
        return format_response(200, result)
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        return format_response(400, {"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return format_response(500, {"error": "Internal server error"})

def format_response(status_code, body):
    """Format the API Gateway response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body)
    }
