FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install --upgrade pip

ENV PROJECT_DIR /app

RUN mkdir $PROJECT_DIR
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR $PROJECT_DIR