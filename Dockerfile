# Use official lightweight Python image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (make sure to have requirements.txt in your root)
COPY requirements.txt .

# Upgrade pip and install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of the project files
COPY . .

# Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
