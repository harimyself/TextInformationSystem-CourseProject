FROM python:3.8
RUN mkdir -p /srv/mainapp
WORKDIR /srv/mainapp
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD []
