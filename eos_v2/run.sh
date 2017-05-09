#!/bin/bash

spk=4264

verdi run eos_v2.py --structure_pk=$spk --group=aaa2 --atomic_files=SSSP_acc_PBE_fixed --kmesh 8 8 8 --partition=cpu --ranks_kp=4 --ranks_diag=36 --ranks_per_node=36
verdi run eos_v2.py --structure_pk=$spk --group=aaa2 --atomic_files=SSSP_acc_PBE_fixed --kmesh 8 8 8 --partition=gpu --ranks_kp=4 --ranks_diag=1 --ranks_per_node=1