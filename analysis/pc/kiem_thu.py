"""Kiem thu tu dong cho thong_ke.py tren du lieu gia lap co dap an biet truoc (Muc 5.8).
Chay: python kiem_thu.py   -> in DAT/KHONG DAT cho tung kiem thu; thoat ma 1 neu co loi."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import thong_ke as tk  # noqa: E402


class BangGia:
    """Bang gia lap: AP cua mot da tap anh = trung binh 'dong gop' cua tung anh (x100)."""

    def __init__(self, dong_gop):
        self.v = np.asarray(dong_gop, float)
        self.img_ids = list(range(len(self.v)))

    def ap(self, idx=None):
        x = self.v if idx is None else self.v[idx]
        return 100 * x.mean(), 100 * x.mean()


def tao(n_anh=300, muc=0.4, lech=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return {s: BangGia(np.clip(rng.normal(muc + 0.02 * s, 0.25, n_anh) + lech, 0, 1)) for s in range(5)}


loi = 0


def kiem(ten, dieu_kien):
    global loi
    print(("DAT       " if dieu_kien else "KHONG DAT ") + ten)
    loi += (not dieu_kien)


# 1. Hai cau hinh giong het -> uoc luong 0, KTC chua 0
a = tao()
d, lo, hi = tk.bootstrap_ghep_cap(a, a, "delta", B=300)
kiem("giong het: delta = 0 va KTC chua 0", abs(d) < 1e-9 and lo <= 0 <= hi)

# 2. Cong them hang so -> KTC khong chua 0 va dung dau
b = {s: BangGia(np.clip(a[s].v + 0.05, 0, 1)) for s in a}
d, lo, hi = tk.bootstrap_ghep_cap(a, b, "delta", B=300)
kiem(f"lech +5 diem: delta={d:.2f}, KTC [{lo:.2f},{hi:.2f}] > 0", lo > 0 and 3 < d < 6)

# 3. R cua chinh no = 1
d, lo, hi = tk.bootstrap_ghep_cap(a, a, "R", B=200)
kiem("R(A,A) = 1", abs(d - 1) < 1e-9)

# 4. Nhieu doc lap cung phan bo (khong co hieu ung that) -> KTC chua 0 phan lon cac lan (~95%)
phu = 0
for r in range(40):
    x, y = tao(seed=100 + r), tao(seed=200 + r)
    y = {s: BangGia(x[s].v * 0 + y[s].v) for s in x}
    _, lo, hi = tk.bootstrap_ghep_cap(x, y, "delta", B=200, seed=r)
    phu += lo <= 0 <= hi
kiem(f"do phu KTC khi khong co hieu ung: {phu}/40 (ky vong ~38)", phu >= 34)

# 5. Ba trang thai: roi nhau va phu kin tren luoi
ok = True
for diem, lo, hi in itertools.product(np.linspace(-1, 1, 21), repeat=3):
    if lo > hi or not (lo <= diem <= hi):
        continue
    t = tk.ba_trang_thai(diem, lo, hi)
    duong = lo > 0 and diem >= tk.DELTA_R
    am = hi < tk.DELTA_R
    ok &= (t in ("Duong", "Am", "Khong ro")) and not (duong and am)
kiem("ba trang thai roi nhau va phu kin", ok)

# 6. Bang quyet dinh phu kin 81 to hop
quy_tac = set()
for combo in itertools.product(["Duong", "Am", "Khong ro"], repeat=4):
    q, _, _ = tk.bang_quyet_dinh(dict(zip("TDNK", combo)))
    quy_tac.add(q)
kiem(f"bang quyet dinh: 81 to hop deu co quy tac ({sorted(quy_tac)})", quy_tac <= {"Q1", "Q2", "Q3", "Q4", "Q5"})
kiem("Q0 khi khong tai lap sup do", tk.bang_quyet_dinh(dict(zip("TDNK", ["Duong"] * 4)), c0_sup_do=False)[0] == "Q0")

# 7. delta_rel tu CV
dr, cv, _ = tk.delta_rel_tu_cv({"x": [10, 10, 10.0001], "y": [20, 20, 20.0001]})
kiem("delta_rel co san 5% khi dao dong rat nho", abs(dr - 0.05) < 1e-9)
dr, cv, _ = tk.delta_rel_tu_cv({"x": [8, 10, 12], "y": [18, 20, 22]})
kiem(f"delta_rel = 2*CV khi CV lon (CV={cv:.3f}, delta_rel={dr:.3f})", abs(dr - 2 * cv) < 1e-9)

# 8. Ket luan tong hop
kiem("tong hop: 2/3 cai thien, 0 kem -> Cai thien",
     tk.ket_luan_tong_hop(["Cai thien", "Cai thien", "Khong khac biet ro"]) == "Cai thien")
kiem("tong hop: co kem hon -> Danh doi",
     tk.ket_luan_tong_hop(["Cai thien", "Kem hon", "Cai thien"]) == "Danh doi")

print("TONG KET:", "TAT CA DAT" if loi == 0 else f"{loi} KIEM THU KHONG DAT")
sys.exit(1 if loi else 0)
