FROM rapidsai/rapidsai:0.12-cuda10.1-runtime-ubuntu18.04
RUN source activate rapids && pip install -U grpcio>=1.24.3 tf-nightly
WORKDIR /home
COPY expt.py /home/
