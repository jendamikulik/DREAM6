import sys, time, json
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('adaptive', 'DREAM6_ZR_ORDER_MEMORY_ADAPTIVE_v03.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['adaptive'] = mod
spec.loader.exec_module(mod)

import numpy as np

known_K1_r5 = 0.7007270812777642  # already exactly verified earlier
results = {}

for K in [2, 3]:
    t0 = time.time()
    dag = mod.build_signature_dag(5, K)
    chart = mod.build_chart(8)
    print(f"K={K}: states={len(dag.nodes)} internals={len(dag.internals)}  build={time.time()-t0:.1f}s", flush=True)

    full_support = np.broadcast_to(
        chart.valid[None], (len(dag.internals),) + chart.valid.shape
    ).copy()

    res, lpm, secs = mod.exactify_once(dag, chart, full_support, 1e-10, None)
    print(f"K={K}: success={res.success}  message={res.message}", flush=True)
    entry = {"K": K, "success": bool(res.success), "message": str(res.message),
             "solve_seconds": float(secs), "variables": int(lpm["nv"])}
    if res.success:
        entry["objective"] = float(res.fun)
        entry["consistent_with_K1"] = bool(res.fun <= known_K1_r5 + 1e-9)
        print(f"K={K}: TRUE unrestricted optimum = {res.fun:.15f}   "
              f"(K=1 known = {known_K1_r5:.15f})   "
              f"consistent (<=K1)? {entry['consistent_with_K1']}   "
              f"solve_time={secs:.1f}s", flush=True)
    results[str(K)] = entry
    with open("exact_r5_K23_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"K={K}: saved to exact_r5_K23_results.json (elapsed so far {time.time()-t0:.1f}s)\n", flush=True)

print("DONE", json.dumps(results, indent=2))
