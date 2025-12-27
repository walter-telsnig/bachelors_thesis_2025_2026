FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for sumy (punkt and punkt_tab for newer nltk)
RUN python -m nltk.downloader punkt punkt_tab

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
