import sys
import os
import base64
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_processor import DocumentProcessor

def test_document_extraction():
    """Test document extraction with a sample document"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if Qwen model path is set (optional)
    qwen_model_path = os.environ.get("QWEN_LOCAL_MODEL_PATH", "models/Qwen2.5-VL-7B-Instruct")
    print(f"Using Qwen model path: {qwen_model_path}")
    print("If the model is not found locally, it will be downloaded from Hugging Face.")
    print("This may take some time for the first run.")
    
    # Create models directory if it doesn't exist
    os.makedirs(qwen_model_path, exist_ok=True)
    
    # Path to test document
    document_path = input("Enter path to test document (PDF or image): ")
    document_type = input("Enter document type (application/pdf, image/jpeg, image/png): ")
    prompt = input("Enter your prompt (e.g., 'What is the invoice number?'): ")
    
    # Read document
    with open(document_path, "rb") as f:
        document_bytes = f.read()
    
    # Process document with Qwen model
    print("\nInitializing Qwen2.5-VL-7B-Instruct model...")
    processor = DocumentProcessor(use_local_model_first=True)
    
    print(f"Processing document with Qwen2.5-VL-7B-Instruct model...")
    result = processor.process_document(document_bytes, document_type, prompt)
    
    # Print result
    print("\nRESULT:")
    print(json.dumps(result, indent=2))
    
    # Simulate Lambda input
    document_base64 = base64.b64encode(document_bytes).decode('utf-8')
    lambda_event = {
        "body": json.dumps({
            "document": document_base64,
            "document_type": document_type,
            "prompt": prompt
        })
    }
    
    # Save Lambda event for future testing
    with open("lambda_test_event.json", "w") as f:
        json.dump(lambda_event, f, indent=2)
    
    print("\nSaved Lambda test event to lambda_test_event.json")
    print("You can use this file to test your Lambda function in the AWS console.")

if __name__ == "__main__":
    test_document_extraction()
