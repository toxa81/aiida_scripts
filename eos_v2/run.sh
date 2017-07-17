#!/bin/bash

spk=4763
grp='many_EoS_v2'

echo $grp

# QE, CPU
verdi run eos_v2.py --group=$grp --kmesh 8 8 8 --partition=cpu --ranks_kp=1 --ranks_diag=36 --ranks_per_node=36 --code=pw.sirius.x@piz_daint $spk

# QE, GPU
verdi run eos_v2.py --group=$grp --kmesh 8 8 8 --partition=gpu --ranks_kp=1 --ranks_diag=1 --ranks_per_node=1 --code=pw.sirius.x@piz_daint $spk

# Exciting
verdi run eos_v2.py --group=$grp --kmesh 8 8 8 --partition=cpu --ranks_kp=36 --ranks_diag=1 --ranks_per_node=36 --code=exciting@piz_daint $spk
