#!/usr/bin/env bash
set -e
npm install -g @openneuro/cli@2.0.1
openneuro download --snapshot 2.0.1 ds003171 ./data/ds003171
git clone https://github.com/MelbourneHci/MelbournePropofolData.git ./data/melbourne
