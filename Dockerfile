# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file
COPY requirements.txt ./requirements.txt

# Install all the necessary libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# The command to start your Gradio app
CMD ["python", "app.py"]
