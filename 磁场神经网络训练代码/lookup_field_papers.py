import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

out=Path('D:/mag/训练结果/field_mlp_v1_analysis/literature')
out.mkdir(parents=True,exist_ok=True)
requests=[
 ('europepmc','https://www.ebi.ac.uk/europepmc/webservices/rest/search',dict(query='"magnetic field" AND ("neural network" OR "deep learning") AND (coil OR actuation) AND (prediction OR modeling OR modelling)',format='json',pageSize=20,resultType='core')),
 ('crossref','https://api.crossref.org/works',{'query':'neural network magnetic field modeling electromagnetic actuation', 'rows':12}),
]
for name,base,params in requests:
    url=base+'?'+urlencode(params)
    result=subprocess.run(['curl.exe','-L','-sS','--max-time','45',url],capture_output=True)
    (out/(name+'.json')).write_bytes(result.stdout)
    (out/(name+'_provenance.json')).write_text(json.dumps(dict(access_date='2026-10-23',url=url,parameters=params,exit_code=result.returncode,error=result.stderr.decode(errors='replace')),ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        payload=json.loads(result.stdout)
        if name=='europepmc':
            assert 'errCode' not in payload and 'resultList' in payload,payload
            print(name,'reported_hits',payload['hitCount'])
            for row in payload['resultList']['result']:
                print(json.dumps({k:row.get(k) for k in ('title','authorString','pubYear','doi','pmcid','abstractText')},ensure_ascii=False))
        else:
            assert payload.get('status')=='ok',payload
            print(name,'reported_hits',payload['message']['total-results'])
            for row in payload['message']['items']:
                print(json.dumps({k:row.get(k) for k in ('title','DOI','published','abstract','link')},ensure_ascii=False))
    except Exception as exc:
        print(name,'FAILED',str(exc)[:1000],result.stdout[:300])
