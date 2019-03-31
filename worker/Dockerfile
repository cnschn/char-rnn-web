FROM tensorflow/tensorflow:1.13.1-py3

RUN pip install keras

ENV PYTHONUNBUFFERED=1

COPY src/ /src/

RUN mkdir /session
WORKDIR /session

CMD ["python", "/src/train.py", "--input", "data.txt", "--freq", "1"]
