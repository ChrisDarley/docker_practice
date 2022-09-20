FROM alpine:latest
FROM python:latest

WORKDIR /files

ADD requirements.txt .
ADD train.py .
ADD inference.py .
ADD main.py .

RUN pip install -r requirements.txt 

CMD ["python", "./main.py"]


