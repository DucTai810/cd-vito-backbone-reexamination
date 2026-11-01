"""Kiem moi module import o muc tren cung cua cac cong cu tinh prototype co san trong env hay khong."""
import ast
import importlib.util
import os
import sys

R = os.path.expanduser("~/CDFSOD-benchmark")
sys.path.insert(0, R)
thieu = set()
for tep in ["tools/extract_instance_prototypes.py", "tools/run_sinkhorn_cluster.py"]:
    cay = ast.parse(open(os.path.join(R, tep)).read())
    for nut in ast.walk(cay):
        ten = []
        if isinstance(nut, ast.Import):
            ten = [a.name for a in nut.names]
        elif isinstance(nut, ast.ImportFrom) and nut.module and nut.level == 0:
            ten = [nut.module]
        for t in ten:
            goc = t.split(".")[0]
            if importlib.util.find_spec(goc) is None:
                thieu.add(goc)
print("THIEU:", sorted(thieu) if thieu else "khong")
