"""
Custom exceptions for the Document AI extraction system
"""

class DocumentProcessingError(Exception):
    """Exception raised when document processing fails"""
    pass

class InvalidInputError(Exception):
    """Exception raised when input validation fails"""
    pass

class LLMProcessingError(Exception):
    """Exception raised when LLM processing fails"""
    pass
