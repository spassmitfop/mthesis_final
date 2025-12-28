FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt update && apt install -y nano
RUN apt-get update && apt-get install -y dos2unix

COPY requirements.txt .

COPY ./cleanrl /cleanrl

COPY ./submodules /submodules
RUN pip install -e /submodules/OC_Atari \
 && pip install -e /submodules/HackAtari

RUN pip install -r requirements.txt

#RUN python --version && abc

RUN apt-get update && apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf && \
    rm -rf /var/lib/apt/lists/*




