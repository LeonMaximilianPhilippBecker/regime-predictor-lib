FROM python:3.11-slim

RUN apt-get update && apt-get install -y sqlite3 && apt-get clean

COPY schema.sql /schema.sql

CMD ["sh", "-c", "sqlite3 /db/quant.db < /schema.sql"]
