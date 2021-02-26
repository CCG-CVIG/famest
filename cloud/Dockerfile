FROM python:3.5-stretch
ENV FLASK_APP server.py
ENV FLASK_RUN_HOST 0.0.0.0
#ENV FLASK_RUN_PORT 8000
ENV FLASK_RUN_CERT /opt/sec/fullchain.pem
ENV FLASK_RUN_KEY /opt/sec/privkey.pem
RUN apt-get update && apt-get install -y -qq build-essential
RUN apt-get install -y libgtk-3-dev
# RUN apt-get install libgtk2.0-dev
# RUN apk add --no-cache gcc musl-dev linux-headers
WORKDIR /opt
RUN mkdir sec
COPY ./files/requirements.txt requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
#RUN python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
RUN pip install -r requirements.txt
# CMD ["flask", "run"]
