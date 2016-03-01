#!/bin/bash

source activate  test
git clone https://github.com/cta-observatory/pyhessio
conda build pyhessio
conda install --use-local pyhessio

