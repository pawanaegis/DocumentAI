FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies for Tesseract OCR and Poppler (for PDF processing)
RUN yum update -y && \
    yum install -y amazon-linux-extras && \
    amazon-linux-extras install epel -y && \
    yum install -y \
    tesseract \
    tesseract-langpack-eng \
    poppler \
    poppler-utils \
    gcc \
    python3-devel && \
    yum clean all

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY src/ .

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]
