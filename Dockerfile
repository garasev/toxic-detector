FROM python:3.10-slim
WORKDIR /app
COPY . .

RUN pip install poetry
RUN poetry install --no-root

COPY toxic-detector /app/toxic-detector
CMD ["poetry", "run", "train"]
