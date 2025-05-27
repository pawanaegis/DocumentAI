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
    
    # Check if Mistral API key is set
    if not os.environ.get("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY environment variable is not set.")
        print("Please create a .env file in the project root with your Mistral API key.")
        sys.exit(1)
    
    # Path to test document
    document_path = input("Enter path to test document (PDF or image): ")
    document_type = input("Enter document type (application/pdf, image/jpeg, image/png): ")
    prompt = input("Enter your prompt (e.g., 'What is the invoice number?'): ")
    
    # Read document
    with open(document_path, "rb") as f:
        document_bytes = f.read()
    
    # Process document
    processor = DocumentProcessor()
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
