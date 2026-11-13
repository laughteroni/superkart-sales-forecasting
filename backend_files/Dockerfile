# Lightweight Python base image - same Python version as the Colab runtime that trained the model (3.13)
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Install dependencies first to make use of Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the API code and the serialized model
COPY . .

# The API listens on port 7860
EXPOSE 7860

# Serve the Flask app with gunicorn (4 workers) on all interfaces
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
