# Start from the official, slim Python 3.12 image
# This matches our server and GitHub Actions environment
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file *first*
# This leverages Docker's cache. It won't reinstall packages
# unless the requirements.txt file changes.
COPY requirements.txt .

# Install all the dependencies
# We add imbalanced-learn here just in case it's missed
RUN pip install --no-cache-dir -r requirements.txt && pip install imbalanced-learn

# Copy the rest of your project's code into the box
# The .dockerignore file will skip .git, venv, etc.
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# The default command to run when the container starts.
# This runs your API server.
CMD ["python", "scripts/serve.py"]