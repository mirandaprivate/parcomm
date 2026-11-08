#!/usr/bin/env python3
"""Validate preserved acceptance markers and export separate backend metrics."""
import csv, json, re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
reference=root/'results/reference'
rows=[]
for case in ['8B_8bit','8B_16bit','70B_8bit','70B_16bit']:
    text=(reference/('result_llama'+case+'.log')).read_text()
    assert 'verified_marker=SECTION6_E2E_VERIFIED' in text,case
    def metric(name,unit):
        m=re.search(r'\| '+re.escape(name)+r' \| ([\d.]+) '+re.escape(unit),text)
        assert m,(case,name)
        return float(m.group(1))
    commit=metric('Commit time','s'); opening=metric('Open time','s')
    setup=re.findall(r'^setup_seconds=([\d.]+)$',text,re.M)
    rows.append(dict(backend='Jolt',case=case,mode='original archived configuration',prover_s=commit+opening,commit_s=commit,open_s=opening,verifier_s=metric('Verifier time','s'),peak_rss_gib=metric('Peak RSS','GiB'),prover_setup_s=float(setup[-1]) if setup else '',verifier_setup_s='',precision='rounded original final-results table'))
for record in json.loads((reference/'qafold-measurements.json').read_text(encoding='utf-8-sig')):
    r=record['result'];v=record['verification']
    assert r['verified'] and v['verified'] and r['cache_mode']=='recompute'
    assert r['prover_profile']['native_openings']==1 and v['profile']['native_openings']==1
    assert abs(r['total_prover_seconds']-r['commit_seconds']-r['opening_with_serialization_seconds'])<1e-8
    rows.append(dict(backend='QAFold',case=record['label'].split('-')[0],mode='low memory',prover_s=r['total_prover_seconds'],commit_s=r['commit_seconds'],open_s=r['opening_with_serialization_seconds'],verifier_s=v['verification_with_read_decode_seconds'],peak_rss_gib=record['peak_rss_kib']/2**20,prover_setup_s=r['setup_seconds'],verifier_setup_s=v['setup_seconds'],precision='original JSON measurements; one run'))
with (root/'results/summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
lines=['# Reference results','', 'Jolt and QAFold expose different statement-hiding guarantees; see README.md. Setup is separate. Jolt values come from rounded log tables. No averaging or new timings.','', '| Backend | Case | Prover s | Verifier s | Peak GiB |','|---|---|---:|---:|---:|']
for r in rows: lines.append(f"| {r['backend']} | {r['case']} | {r['prover_s']:.3f} | {r['verifier_s']:.4f} | {r['peak_rss_gib']:.2f} |")
(root/'results/summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
