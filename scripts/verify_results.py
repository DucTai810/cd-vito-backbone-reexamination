"""Check held-out summaries using paired sampling arithmetic (no raw predictions)."""
import csv
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT / 'results/heldout' / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def main():
    splits = read('per_split_statistics.csv')
    summary = read('heldout_statistics.csv')
    assert len(splits) == 15 and len(summary) == 3
    for row in summary:
        selected = [x for x in splits if x['dataset'] == row['dataset']]
        assert len(selected) == 5 and {int(x['split']) for x in selected} == set(range(5))
        a = [float(x['baseline_AP']) for x in selected]
        b = [float(x['final_AP']) for x in selected]
        relative = [(y-x)/x for x, y in zip(a, b)]
        values = {'baseline_mean': mean(a), 'final_mean': mean(b),
                  'baseline_SD': stdev(a), 'final_SD': stdev(b),
                  'delta_AP': mean(y-x for x, y in zip(a, b)),
                  'delta_relative_percent': 100*mean(relative)}
        for key, value in values.items():
            assert math.isclose(value, float(row[key]), rel_tol=1e-9, abs_tol=1e-9), (row['dataset'], key)
        for x in selected:
            assert math.isclose(float(x['delta_AP']), float(x['final_AP'])-float(x['baseline_AP']), abs_tol=1e-9)
        assert row['decision'] == ('Improved' if row['dataset'] == 'UODD' else 'Inconclusive')
        print(f"{row['dataset']}: {mean(a):.2f} -> {mean(b):.2f}; paired change {100*mean(relative):+.2f}%; {row['decision']}")
    print('Saved mean/SD and paired-change summaries verified; image-bootstrap intervals not recomputed.')

if __name__ == '__main__':
    main()
