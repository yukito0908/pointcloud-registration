# これらのうち 1 行だけコメントアウトしない
# FROM python:3.12.3-bullseye
# FROM python:3.11.9-bullseye
FROM python:3.10.14-bullseye
# FROM python:3.9.19-bullseye
# FROM python:3.8.19-bullseye

RUN pip install poetry
RUN poetry config virtualenvs.create false

WORKDIR /code
#追加
COPY ./submissions/pyproject.toml /code/pyproject.toml
COPY ./submissions/poetry.lock /code/poetry.lock
RUN poetry self add poetry-plugin-export

RUN poetry export --without-hashes --dev --output requirements.txt
RUN pip install -r requirements.txt
#追加
COPY ./submissions/src/pc_reg /code/pc_reg

COPY . /code/
ENV PYTHONPATH=/code
#Docker上でアルゴリズム課題用のmain.pyを実行するエントリポイント
CMD ["python", "-m", pc_reg.main.py"]
