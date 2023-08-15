# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    sox \
    libsox-fmt-mp3 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit and cuDNN (if required)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y nvidia-cudnn

# Copy the application code to the container
COPY . .

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose port 8090
EXPOSE 8090

# Command to run the Flask app on port 8090
CMD ["flask", "run", "--host=0.0.0.0", "--port=8090"]

