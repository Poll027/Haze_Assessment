# Use an official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required by OpenCV (Updated for Debian Trixie/Bookworm)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your actual application code into the container
COPY . /code

# Command to run the FastAPI app. 
# IMPORTANT: Hugging Face Docker Spaces MUST run on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]