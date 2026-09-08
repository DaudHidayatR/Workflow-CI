FROM python:3.12.7-slim-bookworm
RUN printf 'precedence ::ffff:0:0/96 100\n' >> /etc/gai.conf
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV LOGNAME=learner USER=learner PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /workspace
COPY MLProject/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check
COPY . .
CMD ["mlflow", "run", "MLProject", "--env-manager", "local", "--experiment-name", "MSML_DaudHidayatRamadhan"]
