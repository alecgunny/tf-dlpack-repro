FROM rapidsai/rapidsai:0.13-cuda10.1-runtime-ubuntu18.04-py3.6
RUN source activate rapids && pip install -U grpcio>=1.24.3 tensorflow==2.2.0 torch torchvision
WORKDIR /home
COPY expt.py /home/
