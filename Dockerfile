# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the data directory
RUN mkdir -p /app/data

# Download the model and data files from your GitHub repo
RUN curl -LJO https://github.com/emanny0018/ml-project/raw/main/data/fe_combined_matches.csv -o /app/data/fe_combined_matches.csv
RUN curl -LJO https://github.com/emanny0018/ml-project/raw/main/data/voting_classifier.pkl -o /app/data/voting_classifier.pkl

# Copy the prediction script into the container
COPY scripts/predict_match.py /app/predict_match.py

# Make the prediction script executable
RUN chmod +x /app/predict_match.py

# Set the entrypoint to run the prediction script
ENTRYPOINT ["python", "/app/predict_match.py"]
