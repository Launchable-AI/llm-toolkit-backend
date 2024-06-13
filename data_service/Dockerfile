# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install poppler-utils for ocr extraction w/ unstructured
# 2024-02-13 - Trying to remove bloat atm
#RUN apt update && apt install -y libpoppler-dev poppler-utils tesseract-ocr libmagic-dev libreoffice pandoc

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Add the current directory contents into the container at /app
COPY ./src/ /app/src/
COPY main.py /app/

# Expose port 8000 for the app
EXPOSE 8000

# Define the command to run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]