#!/bin/bash
# torch-free interpreter for probe.py + sandbox worker nodes (tools then go through the spool to the local broker, as on the Mac)
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000; L=/leonardo_scratch/large/userexternal/rqian000
E=$F/envs/sc-run
[ -d $E ] || /usr/bin/python3.11 -m venv $E
. $E/bin/activate
pip install -q -U pip 2>&1 | tail -n 1
pip install -q -r $F/sc-tools/req_harness_u.txt 2>&1 | tail -n 3
python -c "import numpy, scipy, sklearn, pandas, yaml; print('run-env ok', numpy.__version__, sklearn.__version__)"
python -c "import importlib.util as u; print('torch present:', u.find_spec('torch') is not None)"
echo RUN_ENV_DONE
