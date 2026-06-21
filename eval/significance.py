"""
시스템별 점수의 신뢰구간 + 시스템 간 차이의 paired bootstrap 유의성 검정.

results/scored_*.jsonl 의 per-item final_score_100 을 사용.
같은 100개 테스트 문항을 공유하므로 paired bootstrap 가능(문항 인덱스로 정렬).

  python eval/significance.py --metric final_score_100
출력: results/significance.json + 콘솔 표
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load(metric):
    data = {}
    for fp in sorted(config.RESULTS_DIR.glob("scored_*.jsonl")):
        rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        name = fp.stem.replace("scored_", "")
        # id 기준 정렬(공유 문항 정렬용). id 없으면 순서 사용
        rows.sort(key=lambda r: str(r.get("id", "")))
        data[name] = {r.get("id", i): r[metric] for i, r in enumerate(rows)}
    return data


def ci(vals, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(vals.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_test(a, b, n_boot=10000, seed=0):
    """공유 문항에 대한 paired bootstrap. p = P(diff<=0) 양측 근사."""
    rng = np.random.default_rng(seed)
    keys = [k for k in a if k in b]
    da = np.array([a[k] for k in keys], float)
    db = np.array([b[k] for k in keys], float)
    diff = da - db
    obs = diff.mean()
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    # 양측 p값: 0을 기준으로 한 부트스트랩 분포의 꼬리
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(obs), float(min(1.0, p)), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="final_score_100")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    data = load(args.metric)
    names = list(data)
    print(f"=== {args.metric}: 평균 ± 95% CI (bootstrap) ===")
    summary = {"metric": args.metric, "systems": {}, "pairwise": []}
    for n in names:
        m, lo, hi = ci(list(data[n].values()), args.boot)
        summary["systems"][n] = {"mean": round(m, 2), "ci95": [round(lo, 2), round(hi, 2)]}
        print(f"  {n:28s} {m:6.2f}  [{lo:.2f}, {hi:.2f}]")

    print("\n=== pairwise paired bootstrap (Δ = A-B, p양측) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            obs, p, k = paired_test(data[a], data[b], args.boot)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            summary["pairwise"].append(
                {"A": a, "B": b, "delta": round(obs, 2), "p": round(p, 4), "n": k, "sig": sig})
            print(f"  {a:24s} vs {b:24s} Δ={obs:6.2f}  p={p:.4f} {sig} (n={k})")

    out = config.RESULTS_DIR / "significance.json"
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
