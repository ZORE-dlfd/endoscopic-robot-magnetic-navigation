from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOIS = ["10.1109/TIE.2024.3426055", "10.1109/TRO.2010.2073030",
        "10.1038/s42256-020-00231-9", "10.1016/j.jcp.2018.10.045",
        "10.1016/j.gie.2010.09.004", "10.1016/S1470-2045(14)70020-8"]


def main():
    records = []
    for doi in DOIS:
        request = urllib.request.Request("https://api.crossref.org/works/" + doi,
                                         headers={"User-Agent": "ResearchMethodsEvidence/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            message = json.load(response)["message"]
        records.append({key: message.get(key) for key in ["DOI", "title", "author", "container-title",
                                                          "volume", "issue", "page", "published", "URL"]})
    primary = Path.home()/"Desktop/Magnetic_Field-Priority_Force_Control_for_Automated_Manipulation_in_Large_Workspaces_With_Reconfigurable_Electromagnetic_Actuation_System.pdf"
    payload = {"retrieved": "2026-10-18", "records": records,
               "local_primary_paper": {"path": str(primary), "sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
                                       "pages": 11, "read_sections": "Title, abstract, III (field-priority configuration optimization), IV (experiments)",
                                       "scope": "Three independently mobile coils and manipulability-aware multiobjective pose optimization; not the same implementation as our fixed-relative eight-source array"},
               "fno_primary": {"url": "https://arxiv.org/abs/2010.08895", "title": "Fourier Neural Operator for Parametric Partial Differential Equations",
                               "venue": "ICLR 2021", "scope": "Future work only; no FNO implementation or benchmark in this project"},
               "fold_blind_spot_sources": [
                   {"doi": "10.1016/j.gie.2010.09.004", "pubmed": "https://pubmed.ncbi.nlm.nih.gov/21067735/",
                    "study": "TERRACE randomized multicenter tandem colonoscopy trial",
                    "basis": "Primary study abstract identifies adenomas hidden behind folds or inner curves and evaluates an additional retrograde view."},
                   {"doi": "10.1016/S1470-2045(14)70020-8", "pubmed": "https://pubmed.ncbi.nlm.nih.gov/24560453/",
                    "study": "International multicentre randomized tandem comparison of forward-viewing and full-spectrum colonoscopy",
                    "basis": "Primary trial evaluates wider mucosal visualization; used as clinical background, not as a performance claim for this capsule."}
               ],
               "clinical_scope": "Targeted primary-literature check, not a systematic review. No study-specific detection benefit is transferred to the local simulation, and no current clinical efficacy claim is made.",
               "citation_policy": "Literature supports background or design principles; all project parameters and results come from local code and data, not from the cited papers."}
    (ROOT/"audit/references_verified.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),"utf-8")
    print("Verified bibliographic records:", len(records))


if __name__ == "__main__":
    main()
