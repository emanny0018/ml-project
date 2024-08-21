# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and data files from your local machine to the container
COPY data/fe_combined_matches.csv /app/fe_combined_matches.csv
COPY data/voting_classifier.pkl /app/voting_classifier.pkl

# Copy the prediction script into the container
COPY scripts/predict_match.py /app/predict_match.py

# Make the prediction script executable
RUN chmod +x /app/predict_match.py

# Set the entrypoint to run the prediction script
ENTRYPOINT ["python", "/app/predict_match.py"]
