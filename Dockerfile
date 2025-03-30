FROM nvcr.io/nvidia/paddlepaddle:24.12-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app

EXPOSE 50505

ENTRYPOINT ["gunicorn", "app:app"]