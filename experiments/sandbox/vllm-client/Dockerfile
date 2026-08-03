FROM vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f

COPY tokenpower_marker.py /opt/tokenpoweragent/tokenpower_marker.py
COPY patch_vllm_benchmark.py /tmp/patch_vllm_benchmark.py

RUN python3 /tmp/patch_vllm_benchmark.py \
    && rm /tmp/patch_vllm_benchmark.py

ENV PYTHONPATH=/opt/tokenpoweragent
