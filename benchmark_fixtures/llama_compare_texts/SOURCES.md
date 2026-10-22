# Source texts for the codec comparison

Same documents and roles as the SmolLM2-135M study (`scripts/bench_llama_compare.py`):

| File | Role | Source |
| --- | --- | --- |
| pep-0484.rst | calibration | PEP 484, Python Enhancement Proposals (public domain) |
| code_typing.py.txt, code_inspect.py.txt, code_subprocess.py.txt, code_tarfile.py.txt, code_logging.py.txt | calibration | CPython standard library modules (PSF License Version 2) |
| code_dataclasses.py.txt | scales (spread fitting) | CPython standard library, dataclasses.py (PSF License Version 2) |
| pep-0008.rst | test (text) | PEP 8 (public domain) |
| code_argparse.py.txt | test (code) | CPython standard library, argparse.py (PSF License Version 2) |

The CPython files are unmodified copies distributed under the Python Software Foundation License Version 2
(https://docs.python.org/3/license.html). Copyright © 2001 Python Software Foundation; All Rights Reserved.
