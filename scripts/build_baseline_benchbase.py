"""Separate measured-success jar with per-transaction HTAP statistics."""
from pathlib import Path
import copy
import build_measured_benchbase as builder

ORIGINAL_PATCHES = copy.deepcopy(builder.PATCHES)


def build():
    builder.BUILD = builder.ROOT / '.runtime/benchbase-baselines'
    builder.JAR = builder.BUILD / 'benchbase-baselines.jar'
    patches = copy.deepcopy(ORIGINAL_PATCHES)
    worker_field = '''private long measuredSuccess = 0;
  private final java.util.Map<String, java.util.List<Long>> measuredLatencies = new java.util.HashMap<>();
  public java.util.Map<String, java.util.List<Long>> getMeasuredLatencies() { return measuredLatencies; }

  public final long getMeasuredSuccess() { return measuredSuccess; }

  private final AtomicInteger intervalRequests'''
    patches['api/Worker.java'][0] = (patches['api/Worker.java'][0][0], worker_field)
    patches['api/Worker.java'][2] = (patches['api/Worker.java'][2][0], '''              intervalRequests.incrementAndGet();
              long successful = txnSuccess.getSampleCount() - successesBefore;
              measuredSuccess += successful;
              if (successful > 0) {
                measuredLatencies.computeIfAbsent(transactionType.getName(), k -> new java.util.ArrayList<>())
                  .add((end - start) / 1000);
              }''')
    before, after = patches['Results.java'][0]
    after = after.replace('  private final State state;', '''  private final java.util.Map<String, java.util.List<Long>> measuredLatencies = new java.util.HashMap<>();
  public void addMeasuredLatencies(java.util.Map<String, java.util.List<Long>> values) {
    values.forEach((k,v) -> measuredLatencies.computeIfAbsent(k, x -> new java.util.ArrayList<>()).addAll(v));
  }
  public java.util.Map<String, Object> measuredTypeStatistics() {
    java.util.Map<String, Object> out = new java.util.TreeMap<>();
    measuredLatencies.forEach((k,v) -> {
      java.util.Collections.sort(v);
      java.util.Map<String,Object> row = new java.util.HashMap<>();
      row.put("count", v.size());
      row.put("throughput_per_second", v.size() / (double) nanoseconds * 1e9);
      row.put("p95_latency_ms", v.get(Math.max(0, (int)Math.ceil(v.size()*.95)-1))/1000.0);
      out.put(k,row);
    });
    java.util.List<Long> analytic = new java.util.ArrayList<>();
    measuredLatencies.forEach((k,v) -> { if (k.matches("Q[0-9]+")) analytic.addAll(v); });
    if (!analytic.isEmpty()) {
      java.util.Collections.sort(analytic);
      java.util.Map<String,Object> row = new java.util.HashMap<>();
      row.put("count",analytic.size());
      row.put("p95_latency_ms",analytic.get((int)Math.ceil(analytic.size()*.95)-1)/1000.0);
      out.put("_analytical",row);
    }
    return out;
  }
  private final State state;''')
    patches['Results.java'][0] = (before, after)
    before, after = patches['ThreadBench.java'][0]
    patches['ThreadBench.java'][0] = (before, after + '\n        results.addMeasuredLatencies(w.getMeasuredLatencies());')
    before, after = patches['util/ResultWriter.java'][0]
    patches['util/ResultWriter.java'][0] = (before, after + '\n    summaryMap.put("Measured Type Statistics", results.measuredTypeStatistics());')
    builder.PATCHES = patches
    return builder.build()


if __name__ == '__main__':
    print(build())
