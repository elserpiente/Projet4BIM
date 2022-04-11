# syntax=docker/dockerfile:1

FROM python:3.10.2

WORKDIR /../Projet4BIM/App

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "UI", "run"]
