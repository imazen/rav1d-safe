"""Paired median ratios and exact one-sided bootstrap upper bounds.

With an odd number of paired rounds, an empirical-bootstrap median is one
of the observed ratios. Its cumulative probability at order statistic i is
P(Binomial(n, i/n) >= (n+1)/2), so no Monte Carlo approximation is needed.
"""
import argparse
import json
from math import comb
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--name', default='cdf-simd-confirm')
p.add_argument('--candidate', default='cdf-simd')
p.add_argument('--references', nargs='+', default=['baseline', 'upstream'])
a = p.parse_args()
rows = json.loads((a.work_dir / (a.name + '-summary.json')).read_text())
records = []
for row in rows:
    samples = row['samples']
    comparisons = {}
    for reference in a.references:
        assert len(samples[a.candidate]) == len(samples[reference])
        ratios = [x / y for x, y in zip(samples[a.candidate], samples[reference])]
        n = len(ratios)
        assert n >= 5 and n % 2 == 1, 'requires an odd number of paired rounds'
        rank = next(i for i in range(1, n + 1)
                    if sum(comb(n, k) * (i / n)**k * (1 - i / n)**(n - k)
                           for k in range(n // 2 + 1, n + 1)) >= .95)
        comparisons[reference] = dict(
            paired_median_ratio=statistics.median(ratios),
            one_sided_bootstrap_95_upper=sorted(ratios)[rank - 1], all_ratios=ratios)
    records.append({k: row[k] for k in ['input', 'threads', 'instances', 'prime']}
                   | dict(comparisons=comparisons, medians=row['medians']))
    b = comparisons[a.references[0]]
    print(row['input'], row['threads'],
          f"change={100 * (b['paired_median_ratio'] - 1):+.2f}%",
          f"upper={b['one_sided_bootstrap_95_upper']:.4f}")
result = dict(method='Exact empirical-bootstrap median quantile via binomial probabilities; '
                    'paired rounds; no discarded samples.', cells=records)
(a.work_dir / (a.name + '-confidence.json')).write_text(json.dumps(result, indent=2) + '\n')
