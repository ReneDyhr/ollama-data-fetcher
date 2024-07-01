FROM python:3.12

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential curl libc6 libc6-dev

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY fetcher.py .

CMD [ "python", "./fetcher.py" ]