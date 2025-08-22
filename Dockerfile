FROM python:3.10-slim-buster
WORKDIR /application
COPY . /application
RUN sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g; s|security.debian.org|archive.debian.org/|g' /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y
RUN pip install -r requirements.txt
CMD ["python3", "application.py"]