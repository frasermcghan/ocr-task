FROM python:3.12

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq -y update \
    && apt-get -qq -y install \
    tesseract-ocr-*

RUN apt-get -qq -y update \
    && apt-get -qq -y install \
    poppler-utils curl

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install --no-cache-dir -r requirements.txt

# Install and pull ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama serve &
RUN sleep 5 && ollama pull llama3.2

ENTRYPOINT [ "ollama serve & && sleep 5 && /bin/bash" ]