FROM python:3.12-slim

WORKDIR /app

# Install system deps (optional but safe for ML libs)
RUN apt-get update && apt-get install -y build-essential

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Default command (overridden in docker-compose)
CMD ["bash"]