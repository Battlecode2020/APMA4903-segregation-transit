#!/usr/bin/env python3
"""Extract only statewide 2020 layers and retrieve official Census supplements."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import shutil
ROOT=Path(__file__).resolve().parents[1]
URLS={
 'ny2020.pl.zip':'https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/New_York/ny2020.pl.zip',
 'tl_2020_36_tabblock20_updated.zip':'https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip',
 '2020PL_R_import_scripts.zip':'https://www2.census.gov/programs-surveys/decennial/rdo/about/2020-census-program/Phase3/SupportMaterials/2020PL_R_import_scripts.zip',
}
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args()
 raw=ROOT/'data/raw';sup=ROOT/'data/supplements';raw.mkdir(parents=True,exist_ok=True);sup.mkdir(parents=True,exist_ok=True)
 with ZipFile(a.archive) as z:
  for kind in ['tabblock','bg','tract','cousub','county','state']:
   stem=f'tl_2020_36_{kind}20'
   for suffix in ['.shp','.shx','.dbf','.prj','.cpg','.shp.iso.xml','.shp.ea.iso.xml']:
    name=stem+suffix
    if name in z.namelist():
     with z.open(name) as src,(raw/name).open('wb') as dst: shutil.copyfileobj(src,dst)
 for name,url in URLS.items():
  dst=sup/name
  if not dst.exists():
   print('Downloading',name,flush=True)
   temp=dst.with_suffix('.partial')
   with urlopen(url,timeout=120) as src,temp.open('wb') as out:shutil.copyfileobj(src,out)
   with ZipFile(temp) as z:
    if z.testzip() is not None:raise ValueError(f'Corrupt download: {name}')
   temp.rename(dst)
 urban=sup/'urban';urban.mkdir(exist_ok=True)
 with ZipFile(sup/'tl_2020_36_tabblock20_updated.zip') as z:
  for suffix in ['.dbf','.cpg']:
   name='tl_2020_36_tabblock20'+suffix
   with z.open(name) as src,(urban/name).open('wb') as dst:shutil.copyfileobj(src,dst)
 print('Inputs ready. Run scripts/build_graphs.py.')
if __name__=='__main__':main()
