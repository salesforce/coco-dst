# docker build --no-cache  multitasking .
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            git \
            ssh \
        emacs-nox \ 
        vim \
        wget \
        less \ 
        psmisc \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            openssh-server openssh-client \
            tmux

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# fair-seq
RUN pip install transformers
RUN pip install conda-build
RUN pip install scikit-learn
RUN pip install sentencepiece
RUN pip install boto3
RUN pip install regex
RUN pip install tensorboard
RUN pip install future
RUN pip install matplotlib
RUN pip install sacremoses subword_nmt

# additional python packages
RUN conda install tqdm 
RUN conda install pandas
RUN conda install python-dateutil


# from github
# RUN git clone https://github.com/nvidia/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
CMD bash 
