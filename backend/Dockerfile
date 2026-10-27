# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/backend

COPY backend/requirements.txt ./requirements.txt

# Install CUDA-enabled PyTorch first and keep it compatible with later GPU inference.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --retries 10 --timeout 60 --upgrade pip && \
    pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --retries 10 --timeout 60 torch

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --retries 10 --timeout 60 -r requirements.txt

RUN python -c "import torch; assert torch.version.cuda is not None, 'CUDA-enabled PyTorch is required'"

COPY backend/ ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
