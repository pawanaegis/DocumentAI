# Document AI Extraction System

This project implements an AI-powered document extraction system that can process document images and PDFs (invoices, bills, receipts, etc.) and extract specific information based on user prompts.

## Features

- Process document images (PNG, JPEG) and PDFs
- Extract text and structured information from documents
- Answer specific questions about document content
- Deployed as a serverless application on AWS Lambda using ECR

## Architecture

The system uses:
- AWS Lambda for serverless execution
- Amazon ECR for container management
- Document AI processing with OCR and LLM capabilities
- PDF processing libraries

## Setup and Deployment

See the deployment instructions in `deployment/README.md`.

## Usage

The Lambda function accepts requests with:
- Document: Base64 encoded image or PDF
- Prompt: User's question about the document (e.g., "What is the invoice number?")

It returns:
- Extracted information based on the prompt
- Confidence score
- Processing metadata
