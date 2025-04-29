Bootstrap: docker
From: continuumio/miniconda3

%files
  environment.yml /tmp/environment.yml
  .              /workspace

%post
  apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
  curl -fsSL https://deb.nodesource.com/setup_14.x | bash -
  apt-get update && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
  conda env create -f /tmp/environment.yml
  source activate impact-synergy

%environment
  export PATH="/opt/conda/envs/impact-synergy/bin:$PATH"
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8

%runscript
  exec bash "$@"
