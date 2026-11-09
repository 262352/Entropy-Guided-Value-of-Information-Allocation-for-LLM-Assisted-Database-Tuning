#!/usr/bin/env python3
"""Summarize provider-reported prefix hits separately from local reply reuse."""
import argparse
import json
from pathlib import Path


def summarize(records):
    provider=[r for r in records if r.get('origin')=='provider']
    known=[r for r in provider if r.get('cache_hit_tokens') is not None and r.get('cache_miss_tokens') is not None]
    hit=sum(r['cache_hit_tokens'] for r in known)
    miss=sum(r['cache_miss_tokens'] for r in known)
    def total(key):
        values=[r[key] for r in provider if r.get(key) is not None]
        return sum(values) if values else None
    return {'provider_responses':len(provider),
        'local_response_reuses':sum(r.get('origin')=='local_response' for r in records),
        'responses_with_cache_telemetry':len(known),'responses_without_cache_telemetry':len(provider)-len(known),
        'input_tokens_reported':total('input_tokens'),'output_tokens_reported':total('output_tokens'),
        'cache_hit_tokens':hit if known else None,'cache_miss_tokens':miss if known else None,
        'cache_hit_rate_for_reported_responses':hit/(hit+miss) if hit+miss else None}


def report(root):
    files={};records=[]
    for path in sorted(Path(root).rglob('request_metrics.jsonl')):
        contents=path.read_text();lines=contents.splitlines();current=[]
        for i,line in enumerate(lines):
            if not line.strip(): continue
            try: current.append(json.loads(line))
            except json.JSONDecodeError:
                if i==len(lines)-1 and not contents.endswith('\n'): break
                raise
        files[str(path)]=summarize(current);records.extend(current)
    return {'scope':str(Path(root).resolve()),'total':summarize(records),'files':files,
        'note':'Token-weighted service-prefix hit rate, not local response reuse. Missing provider telemetry remains unknown. Transport failures without returned usage cannot be costed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();data=report(args.directory)
    encoded=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded)
    print(encoded,end='')
