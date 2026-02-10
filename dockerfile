# Use a slim Python image for a smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Point both Transformers and NLTK to a local cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV NLTK_DATA=/app/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories
RUN mkdir -p /app/model_cache /app/nltk_data

# Pre-download the L12 model (Matches your service logic)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')"

# Pre-download NLTK stopwords
RUN python -m nltk.downloader stopwords -d /app/nltk_data

# Copy the rest of the application code
COPY . .

# Ensure the app can write the cluster_plot.png to the flat root
RUN chmod 777 /app

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]