#!/usr/bin/env python3
"""Build an isolated BenchBase jar with measurement-window success counting.

The original jar/source are preserved. Only successful operations starting and
ending in the same MEASURE phase contribute, exactly matching the latency
window. Warmup, aborted requests and boundary-straddling commits are excluded.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.runtime/benchbase'
BASE = SOURCE / 'target/benchbase-postgres'
BUILD = ROOT / '.runtime/benchbase-measured'
JAR = BUILD / 'benchbase-measured.jar'

PATCHES = {
 'api/Worker.java': [
    ('private final AtomicInteger intervalRequests', 'private long measuredSuccess = 0;\n\n  public final long getMeasuredSuccess() { return measuredSuccess; }\n\n  private final AtomicInteger intervalRequests'),
    ('        doWork(configuration.getDatabaseType(), transactionType);', '        long successesBefore = txnSuccess.getSampleCount();\n        doWork(configuration.getDatabaseType(), transactionType);'),
    ('              intervalRequests.incrementAndGet();', '              intervalRequests.incrementAndGet();\n              measuredSuccess += txnSuccess.getSampleCount() - successesBefore;'),
 ],
 'Results.java': [
    ('  private final State state;', '  private long measuredSuccess = 0;\n  public void addMeasuredSuccess(long count) { measuredSuccess += count; }\n  public long getMeasuredSuccess() { return measuredSuccess; }\n  public double measuredSuccessPerSecond() { return (double) measuredSuccess / nanoseconds * 1e9; }\n\n  private final State state;'),
 ],
 'ThreadBench.java': [
    ('        results.getSuccess().putHistogram(w.getTransactionSuccessHistogram());', '        results.getSuccess().putHistogram(w.getTransactionSuccessHistogram());\n        results.addMeasuredSuccess(w.getMeasuredSuccess());'),
 ],
 'util/ResultWriter.java': [
    ('    summaryMap.put("Goodput (requests/second)", results.requestsPerSecondGoodput());', '    summaryMap.put("Goodput (requests/second)", results.requestsPerSecondGoodput());\n    summaryMap.put("Measured Successful Requests", results.getMeasuredSuccess());\n    summaryMap.put("Measured Goodput (requests/second)", results.measuredSuccessPerSecond());'),
 ],
}


def patch_source(text, replacements):
    for before, after in replacements:
        if text.count(before) != 1:
            raise RuntimeError('BenchBase source mismatch; refusing an ambiguous metric patch')
        text = text.replace(before, after)
    return text


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    source_dir = SOURCE / 'src/main/java/com/oltpbenchmark'
    inputs = {name: (source_dir / name).read_text() for name in PATCHES}
    signature = hashlib.sha256((json.dumps(PATCHES, sort_keys=True) + json.dumps(inputs, sort_keys=True)
                               + hashlib.sha256((BASE / 'benchbase.jar').read_bytes()).hexdigest()).encode()).hexdigest()
    manifest = BUILD / 'provenance.json'
    if JAR.exists() and manifest.exists() and json.loads(manifest.read_text()).get('signature') == signature:
        if hashlib.sha256(JAR.read_bytes()).hexdigest() == json.loads(manifest.read_text())['jar_sha256']:
            return JAR
    sources = []
    for name, text in inputs.items():
        path = BUILD / 'sources/com/oltpbenchmark' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch_source(text, PATCHES[name]))
        sources.append(str(path))
    classes = BUILD / 'classes'
    if classes.exists():
        shutil.rmtree(classes)
    classes.mkdir()
    classpath = str(BASE / 'benchbase.jar') + ':' + str(BASE / 'lib/*')
    subprocess.run(['javac', '--release', '21', '-cp', classpath, '-d', str(classes), *sources], check=True)
    shutil.copy2(BASE / 'benchbase.jar', JAR)
    subprocess.run(['jar', '--update', '--file', str(JAR), '-C', str(classes), '.'], check=True)
    # The jar manifest resolves dependencies relative to the jar's location.
    if not (BUILD / 'lib').exists():
        (BUILD / 'lib').symlink_to(BASE / 'lib', target_is_directory=True)
    manifest.write_text(json.dumps({'signature': signature, 'jar_sha256': hashlib.sha256(JAR.read_bytes()).hexdigest(),
        'upstream_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'definition': 'successful operations wholly within one MEASURE phase / measurement elapsed seconds',
        'patched_sources': sorted(PATCHES)}, indent=2))
    return JAR


if __name__ == '__main__':
    print(build())
