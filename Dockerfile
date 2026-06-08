# 1. Start with a lightweight, official Python base operating system
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install the system-level C-libraries required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1

# 4. Copy our requirements file into the container
COPY requirements.txt .

# 5. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy our necessary code and model directories
COPY src/ ./src/
COPY models/ ./models/

# 7. Expose the port
EXPOSE 8000

# 8. Boot the server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
