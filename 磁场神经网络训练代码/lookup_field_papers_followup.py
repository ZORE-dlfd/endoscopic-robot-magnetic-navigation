import json, subprocess, sys
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
out=Path('D:/mag/训练结果/field_mlp_v1_analysis/literature')
requests=[
 ('openalex_aim','https://api.openalex.org/works/doi:10.1109/AIM55361.2024.10637101',{}),
 ('epmc_targeted','https://www.ebi.ac.uk/europepmc/webservices/rest/search',{'query':'(TITLE_ABS:"magnetic field") AND (TITLE_ABS:"neural network" OR TITLE_ABS:"deep learning") AND (TITLE_ABS:actuation OR TITLE_ABS:prediction OR TITLE_ABS:modeling) AND FIRST_PDATE:[2015-01-01 TO 2026-10-23]','pageSize':15,'resultType':'core','format':'json'}),
 ('arxiv_targeted','https://export.arxiv.org/api/query',{'search_query':'ti:"magnetic field" AND (abs:"neural network" OR abs:"deep learning")','max_results':12}),
]
def fetch(req):
 name,base,params=req
 url=base+('?' + urlencode(params) if params else '')
 r=subprocess.run(['curl.exe','-L','-sS','--max-time','45','-w','\n%{http_code}',url],capture_output=True)
 body,_,status=r.stdout.rpartition(b'\n')
 (out/(name+'.raw')).write_bytes(body)
 (out/(name+'_provenance.json')).write_text(json.dumps(dict(date='2026-10-23',url=url,params=params,http_status=status.decode(),exit_code=r.returncode),ensure_ascii=False,indent=2),encoding='utf-8')
 if status!=b'200': return name,status.decode(),body[:300].decode(errors='replace')
 if name.startswith('arxiv'): return name,status.decode(),body.decode()
 d=json.loads(body)
 if name.startswith('openalex'):
  inv=d.get('abstract_inverted_index') or {}
  pos={i:w for w,ii in inv.items() for i in ii}
  return name,dict(title=d.get('title'),abstract=' '.join(pos[i] for i in sorted(pos)),locations=d.get('locations'),authors=[a['author']['display_name'] for a in d.get('authorships',[])])
 assert 'resultList' in d and 'errCode' not in d
 return name,d.get('hitCount'),[{k:r.get(k) for k in ('title','doi','pmcid','abstractText','pubYear')} for r in d['resultList']['result']]
with ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(fetch,requests): print(json.dumps(result,ensure_ascii=False))
