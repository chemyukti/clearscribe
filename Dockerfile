# Container for Hugging Face Spaces (Docker SDK); also runs anywhere Docker does.
FROM python:3.11-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small (the default wheel bundles CUDA).
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the speech model while building, so the first visitor does not wait for it.
ARG CLEARSCRIBE_MODEL=facebook/wav2vec2-base-960h
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC as M; \
n='${CLEARSCRIBE_MODEL}'; Wav2Vec2Processor.from_pretrained(n); M.from_pretrained(n)"

COPY --chown=user . .

EXPOSE 7860
# One worker: review state is kept in memory and must be shared by every request.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "600", "app:create_app()"]
