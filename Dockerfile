FROM python:3.12-slim

# Install system dependencies (ffmpeg and curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set application workspace
WORKDIR /app

# Copy requirements and credentials
COPY requirements.txt .env* ./

# Install python dependencies globally inside container
RUN pip install --no-cache-dir -r requirements.txt

# Copy source agent file
COPY agent.py ./

# Pre-create standard mount folders
RUN mkdir -p /input /output

# Default run entrypoint
CMD ["python", "agent.py"]
