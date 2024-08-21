# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the model and data files from your GitHub repo
RUN curl -LJO https://github.com/emanny0018/ml-project/raw/main/data/fe_combined_matches.csv
RUN curl -LJO https://github.com/emanny0018/ml-project/raw/main/data/voting_classifier.pkl

# Copy the prediction script into the container
COPY scripts/predict_match.py .

# Make the prediction script executable
RUN chmod +x predict_match.py

# Run the script with the appropriate environment variables
ENTRYPOINT ["python", "predict_match.py"]
