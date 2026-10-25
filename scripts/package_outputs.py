#!/usr/bin/env python3
"""Bundle graph artifacts, source code, and environment lock without raw data."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'outputs/ny_2020'
assert (folder/'readback_validation.json').exists(), 'Verify outputs before packaging'
with ZipFile(ROOT/'outputs/ny_2020_dual_graphs.zip','w',ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(folder.rglob('*')):
        if f.is_file():z.write(f,Path('ny_2020_dual_graphs/outputs/ny_2020')/f.relative_to(folder))
    for f in [ROOT/'README.md',ROOT/'requirements.txt',*sorted((ROOT/'scripts').glob('*.py'))]:
        z.write(f,Path('ny_2020_dual_graphs')/f.relative_to(ROOT))
print(ROOT/'outputs/ny_2020_dual_graphs.zip')
