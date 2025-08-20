FROM python:3.10-slim-buster
WORKDIR /application
COPY . /application
RUN apt-get update && apt-get upgrade -y
RUN pip install -r requirements.txt
CMD ["python3", "application.py"]