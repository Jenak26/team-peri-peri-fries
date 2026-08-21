# The examination engine, packaged for Hugging Face Spaces.
#
# It runs here rather than on a serverless platform for three concrete reasons: the
# PyTorch install is roughly 1 GB against a 250 MB serverless bundle cap, one
# examination takes minutes against a 60-300 s function timeout, and the pipeline
# shells out to ffmpeg, which serverless Python runtimes do not ship.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000 and only $HOME is writable.
RUN useradd -m -u 1000 peri
USER peri
ENV HOME=/home/peri \
    PATH=/home/peri/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/peri/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=peri requirements-cpu.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-cpu.txt

# Stage B builds its encoder with SegformerForSemanticSegmentation.from_pretrained,
# which reaches the Hub the first time a checkpoint is loaded. Doing that during the
# build means the first upload after a cold start is not waiting on a network fetch,
# and a Hub hiccup cannot fail an examination.
RUN python -c "from transformers import SegformerForSemanticSegmentation as M;     M.from_pretrained('nvidia/mit-b2', num_labels=2, ignore_mismatched_sizes=True)"

COPY --chown=peri . .

# Evidence is examination output and must be writable at runtime.
RUN mkdir -p evidence artifacts

# Spaces routes public traffic to 7860.
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
