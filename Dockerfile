FROM python:3.10-slim
WORKDIR /app
COPY . .

RUN pip install poetry
RUN poetry install

COPY toxic_comment_classification /app/toxic_comment_classification
CMD ["poetry", "run", "train"]
