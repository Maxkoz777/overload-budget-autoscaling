FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /tools
COPY tools/kube.py tools/controller.py tools/collector.py policies.py ./
