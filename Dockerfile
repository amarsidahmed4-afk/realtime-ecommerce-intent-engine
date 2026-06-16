# 1. Use an explicit, lightweight base image to speed up cold-start provisioning
FROM python:3.11-slim

# 2. Set system environment variables to optimize Python performance inside the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 3. Establish the working directory inside the container isolation layer
WORKDIR /app

# 4. Install system dependencies required for minimal C-extensions (like LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Leverage Docker build caching by copying and installing requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy the core source code and serialized model assets into the container image
COPY src/ ./src/
COPY models/ ./models/

# 7. Expose the standard Cloud Run port container interface
EXPOSE 8080

# 8. Launch Uvicorn bound to the dynamic Cloud Run environment port
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT}