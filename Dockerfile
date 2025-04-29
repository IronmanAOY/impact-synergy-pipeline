FROM continuumio/miniconda3

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy
SHELL ["conda","run","-n","impact-synergy","/bin/bash","-lc"]

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get update && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace
ENTRYPOINT ["bash","scripts/run_all.sh"]
