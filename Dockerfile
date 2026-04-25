FROM python:3.11-slim

WORKDIR /app

# Required runtime libraries for OpenCV (cv2) on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --use-pep517 -r requirements.txt

COPY app/main.py ./main.py
COPY model ./model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]