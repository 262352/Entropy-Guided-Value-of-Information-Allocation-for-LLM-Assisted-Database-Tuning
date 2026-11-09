"""Workload plumbing tests without restarting PostgreSQL or calling an LLM."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twitter_xml_uses_same_scale_trace_window_and_private_permissions(tmp_path, monkeypatch):
    h = load_script('real_benchbase')
    config = tmp_path/'config/postgres'
    config.mkdir(parents=True)
    (config/'sample_twitter_config.xml').write_text('''<parameters><url/><username/><password/>
        <scalefactor/><terminals/><isolation/><works><work><time/><rate/>
        <weights>1,1,7,90,1</weights></work></works></parameters>''')
    monkeypatch.setattr(h, 'BENCH', tmp_path)
    monkeypatch.setattr(h, 'RUNTIME', tmp_path)
    monkeypatch.setattr(h, 'private_env', lambda: {'PGPASSWORD':'test-only'})
    output = tmp_path/'twitter.xml'
    h.xml_config('cognitune_twitter', output, 15, 60)
    root = ET.parse(output).getroot()
    assert root.findtext('scalefactor') == '10'
    assert root.findtext('terminals') == '16'
    assert root.findtext('works/work/warmup') == '15'
    assert root.findtext('works/work/time') == '60'
    assert root.findtext('works/work/weights') == '1,1,7,90,1'
    assert root.findtext('randomSeed') == '0'
    assert root.findtext('tracefile') == str(tmp_path/'twitter-traces/tweetids.txt')
    assert output.stat().st_mode & 0o777 == 0o600


def test_measured_metric_patch_fails_closed_for_changed_upstream():
    builder = load_script('build_measured_benchbase')
    with pytest.raises(RuntimeError, match='source mismatch'):
        builder.patch_source('changed upstream source', [('not found','replacement')])
    # Verify actual source compatibility where the runtime is installed.
    source = ROOT/'.runtime/benchbase/src/main/java/com/oltpbenchmark'
    if not source.exists():
        pytest.skip('local BenchBase source not installed')
    for name, replacements in builder.PATCHES.items():
        assert builder.patch_source((source/name).read_text(), replacements)


def test_measured_evaluation_excludes_warmup_and_aborts(tmp_path, monkeypatch, space):
    h = load_script('real_benchbase')
    monkeypatch.setattr(h, 'RUNTIME', tmp_path)
    monkeypatch.setattr(h, 'pgctl', lambda _: None)
    monkeypatch.setattr(h, 'configure', lambda _: None)
    monkeypatch.setattr(h, 'maintenance_sql', lambda *args: None)
    monkeypatch.setattr(h, 'xml_config', lambda *args: None)
    monkeypatch.setattr(h, 'sql', lambda *_: '{"fsync":"on","synchronous_commit":"on","full_page_writes":"on"}')
    monkeypatch.setattr(h, 'benchbase', lambda *args: {
        'Final State':'EXIT', 'Measured Requests':100,
        'Measured Successful Requests':80, 'Measured Goodput (requests/second)':8,
        'Throughput (requests/second)':10, 'Goodput (requests/second)':15})
    evaluator = h.LocalBenchBaseEvaluator(space, tmp_path/'measurements', 4)
    result = evaluator.evaluate(space.get_default(), 0)
    assert result.status == 'SUCCESS' and result.raw_metric == 8
    assert result.metadata['measured_success_fraction'] == .8
    assert result.metadata['metric'] == h.METRIC
    with pytest.raises(FileExistsError):
        evaluator.evaluate(space.get_default(), 0)
