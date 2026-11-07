# Нейросетевой синтез фазового ДОЭ (модель unrolled/, 8 стадий) — запуск на CPU.
#
#   docker build -t doe-nn .
#   docker run --rm -v "$PWD/input:/data/input:ro" -v "$PWD/output:/data/output" doe-nn
#
# input/  — подпапки-комплекты по 4 PNG 128×128 (или сами 4 PNG);
# output/ — для каждого комплекта doe.png (+ восстановления, метрики) и summary.csv.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch — CPU-сборка (без CUDA-библиотек, образ ~1 ГБ вместо ~6 ГБ)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.9.1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY unrolled/ /app/unrolled/

# самопроверка при сборке: модель загружается и считает ДОЭ
RUN python /app/unrolled/predict.py --input /nonexistent --output /tmp/selfcheck --no-extras \
    && rm -rf /tmp/selfcheck

VOLUME ["/data/input", "/data/output"]
ENTRYPOINT ["python", "/app/unrolled/predict.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]
