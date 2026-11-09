import importlib.util
import json
from pathlib import Path
import pytest


spec=importlib.util.spec_from_file_location('ablation_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_twitter_ablation.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_failed_rechecks_do_not_create_a_success_only_median():
    rows=[{'status':'SUCCESS','raw_metric':100},{'status':'CRASH','raw_metric':0}]
    stats=analysis.measurement_stats(rows)
    assert stats['successes']==1 and stats['median'] is None and stats['cv'] is None


def test_audit_keeps_peak_and_validation_distinct_and_accepts_official_diagnostics(tmp_path):
    run=tmp_path/'runs/mctuner_official/seed_0';run.mkdir(parents=True)
    (tmp_path/'protocol.json').write_text(json.dumps({'n_init':1,'budget':2,'final_validation_repeats':3}))
    (tmp_path/'default_before.json').write_text(json.dumps([{'status':'SUCCESS','raw_metric':50}]*3))
    records=[{'trial':i,'method':'mctuner_official','raw_metric':metric,'status':'SUCCESS','candidate_source':'test'}
             for i,metric in enumerate([100,200])]
    (run/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (run/'round_diagnostics.jsonl').write_text('{"trial":1,"nodes":3}\n')
    (run/'final_validation.json').write_text(json.dumps({'results':[{'status':'SUCCESS','raw_metric':v} for v in [130,150,140]]}))
    row,details=analysis.collect('official',run)
    assert row['best_TPS']==200 and row['validation_median_TPS']==140
    assert row['search_gain_over_initial_percent']==100
    assert row['validated_gain_over_default_percent']==pytest.approx(180)
    assert details['curve']==[100,200] and row['active_sets']==0


def test_live_partial_last_line_is_ignored_but_interior_corruption_is_not(tmp_path):
    import pytest
    path=tmp_path/'trace.jsonl';path.write_text('{"trial":0}\n{"trial":')
    assert analysis.read_rows(path)==[{'trial':0}]
    path.write_text('{"trial":0}\nbroken\n{"trial":1}\n')
    with pytest.raises(json.JSONDecodeError): analysis.read_rows(path)
