#!/usr/bin/env bash

python gen_21cm.py
python gen_fg.py
python fix_ra_dec.py
python plot_slice.py
python plot_cl.py
python plot_decomp.py
python plot_pk2d.py
python plot_pk1d.py