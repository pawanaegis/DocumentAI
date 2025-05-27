# Deployment Instructions

This document outlines the steps to deploy the Document AI extraction system to AWS Lambda using Amazon ECR.

## Prerequisites

- AWS CLI installed and configured with appropriate permissions
- Docker installed locally
- An AWS account with access to ECR, Lambda, and IAM services

## Deployment Steps

### 1. Set up environment variables

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### 2. Build and push Docker image to ECR

Run the deployment script:

```bash
./deployment/deploy.sh
```

This script will:
- Create an ECR repository if it doesn't exist
- Build the Docker image
- Tag the image
- Push the image to ECR

### 3. Create Lambda function

After pushing the image to ECR, you can create a Lambda function using the AWS Management Console or AWS CLI:

- Function name: `document-ai-extractor`
- Runtime: Container Image
- Image URI: Select the image you pushed to ECR
- Memory: 2048 MB (recommended minimum)
- Timeout: 30 seconds (adjust based on your document complexity)
- Environment variables:
  - OPENAI_API_KEY: your_openai_api_key

### 4. Set up API Gateway (optional)

If you want to expose your Lambda function as an API:

1. Create a new REST API in API Gateway
2. Create a POST method and integrate it with your Lambda function
3. Deploy the API to a stage (e.g., "prod")
4. Note the API endpoint URL

## Testing

You can test your deployment using the AWS Lambda console or by sending a request to your API Gateway endpoint:

```bash
curl -X POST https://your-api-gateway-url/prod \
  -H "Content-Type: application/json" \
  -d '{
    "document": "base64_encoded_document_here",
    "document_type": "application/pdf",
    "prompt": "What is the invoice number?"
  }'
```
