import json,subprocess,sys
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
out=Path('D:/mag/训练结果/field_mlp_v1_analysis/literature')
jobs=[
 ('crossref_calibration','https://api.crossref.org/works',{'query':'deep neural network magnetic field electromagnetic actuation calibration','rows':10}),
 ('ship_fulltext','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12749233/fullTextXML',{}),
 ('gan_html','https://arxiv.org/html/2203.07897v1',{}),
 ('lpf_html','https://arxiv.org/html/0908.4564v1',{}),
]
def fetch(job):
 name,base,params=job
 url=base+('?' + urlencode(params) if params else '')
 r=subprocess.run(['curl.exe','-L','-sS','--max-time','40','-w','\n%{http_code}',url],capture_output=True)
 body,_,status=r.stdout.rpartition(b'\n')
 (out/(name+'.raw')).write_bytes(body)
 (out/(name+'_provenance.json')).write_text(json.dumps(dict(date='2026-10-23',url=url,params=params,http_status=status.decode(),exit_code=r.returncode),ensure_ascii=False,indent=2),encoding='utf-8')
 if name.startswith('crossref') and status==b'200':
  d=json.loads(body); assert d.get('status')=='ok'
  return name,[{k:a.get(k) for k in ('title','DOI','abstract','published')} for a in d['message']['items']]
 return name,status.decode(),len(body),body[:100].decode(errors='replace')
with ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(fetch,jobs): print(json.dumps(result,ensure_ascii=False))
