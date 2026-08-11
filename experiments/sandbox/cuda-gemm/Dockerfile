FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

WORKDIR /src
COPY gemm.cu /src/gemm.cu
RUN nvcc -O3 -std=c++17 -lineinfo \
    -gencode arch=compute_90,code=sm_90 \
    /src/gemm.cu -lcublas -o /usr/local/bin/tpa-gemm

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

COPY --from=build /usr/local/bin/tpa-gemm /usr/local/bin/tpa-gemm

USER 10001:10001
ENV HOME=/tmp
ENTRYPOINT ["/usr/local/bin/tpa-gemm"]
