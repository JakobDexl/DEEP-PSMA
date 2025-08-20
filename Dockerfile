#FROM --platform=linux/amd64 python:3.11-slim AS example-nntest
FROM --platform=linux/amd64 pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
# import basic python container to ensure pytorch installed with GPU drivers via PIP below

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user pyproject.toml /opt/app/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user weights_single/ /opt/app/weights/
#COPY --chown=user:user weights/ /opt/app/weights/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    -e .

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user inference_single_bothways.py /opt/app/
COPY --chown=user:user inference_single.py /opt/app/
COPY --chown=user:user inference_bothways_ensemble.py /opt/app/

ENTRYPOINT ["python", "inference_single_bothways.py"]
#ENTRYPOINT ["python", "inference_bothways_ensemble.py"]
#ENTRYPOINT ["python", "inference_single.py"]
#ENTRYPOINT ["/bin/bash"] #switch entry point to enable interactive session for testing "-it" flag


