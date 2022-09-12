FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y pytorch torchvision -c pytorch
RUN mkdir /workspace/ && cd /workspace/ && git clone hhttps://github.com/javirk/pytorch-CycleGAN-and-pix2pix.git && cd pytorch-CycleGAN-and-pix2pix && pip install -r requirements.txt

WORKDIR /workspace
