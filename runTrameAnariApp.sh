#!/bin/sh

mpiexec -genv LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tmarrinan/local/lib -genv ANARI_LIBRARY=barney -np 8 python trame_anari.py --host 0.0.0.0 --port 8008 --server --timeout 0

