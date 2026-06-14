FROM python:3.12

WORKDIR /rag

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless tesseract-ocr && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN python -m playwright install --with-deps chromium

COPY rag/lib ./rag/lib

# TODO : COPY chainlit_app ./chainlit_app

EXPOSE 80

# TODO : CMD ["chainlit", "run", "chainlit_app/app.py", "--host", "0.0.0.0", "--port", "80"]

CMD ["python", "--version"]
