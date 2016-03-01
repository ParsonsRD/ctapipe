#!/bin/sh
git clone https://github.com/cta-observatory/pyhessio
cd pyhessio
python setup.py install
python setup.py develop

