# Use Python 3.13
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (better for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Start the app
CMD ["python", "app.py"]
