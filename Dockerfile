# Dockerfile

# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]