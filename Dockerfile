FROM python:3.12-slim

# Install system dependencies (ffmpeg and curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set application workspace
WORKDIR /app

# Copy package configurations and credentials
COPY pyproject.toml README.md .env ./

# Install python dependencies globally inside container
RUN pip install --no-cache-dir -e .

# Copy source tree files
COPY app/ app/
COPY schemas/ schemas/
COPY workers/ workers/
COPY media/ media/
COPY audio/ audio/
COPY vision/ vision/
COPY fusion/ fusion/
COPY llm/ llm/
COPY display/ display/
COPY demo.py ./
COPY .streamlit/ .streamlit/

# Pre-create standard mount folders
RUN mkdir -p /input /output

# Default run entrypoint
CMD ["python", "-m", "app.main"]
