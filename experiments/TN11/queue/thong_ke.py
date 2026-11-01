"""Thong ke va quy tac quyet dinh cua de tai (Muc 3.4, 5.8; Bang 8, Bang 9). Viet TRUOC khi co ket qua.

Doi tuong "bang": moi vat co .img_ids va .ap(chi_so_anh) -> (AP, AP50)  (vd ap_nhanh.BangKhop).
Mot "luot theo lan lay mau" = dict {lan_lay_mau: bang}.
"""
import math

import numpy as np

B_MAC_DINH = 2000
DELTA_R = 0.25               # Bang 8: nguong tac dong cua mot nhom nguyen nhan
DELTA_R_DO_NHAY = (0.15, 0.35)
NGUONG_SUP_DO = 0.2          # R <= 0.2
NGUONG_HOI_PHUC = 0.9        # can duoi KTC cua R >= 0.9
TI_LE_KHONG_TAM_THUONG = 1e-2  # eta_b >= 1e-2 * eta_h


# ----------------------------------------------------------------------------- bootstrap
def _ap(bang, idx):
    return bang.ap(idx)[0]


def bootstrap_ghep_cap(a, b, dai_luong="delta", B=B_MAC_DINH, seed=0):
    """Bootstrap ghep cap hai tang (lan lay mau, anh) giua cau hinh a (moc) va b tren CUNG mot bo.

    a, b: dict {lan_lay_mau: bang}; dung cac lan lay mau chung.
    dai_luong: 'delta' = AP_b - AP_a ; 'delta_rel' = (AP_b - AP_a)/AP_a ; 'R' = AP_b / AP_a.
    Thong ke = trung binh tren cac lan lay mau cua dai luong tinh theo tung lan lay mau.
    Tra ve (uoc luong diem, can duoi, can tren) cua KTC 95% phan vi.
    """
    chung = sorted(set(a) & set(b))
    assert chung, "khong co lan lay mau chung"

    def mot(sa, sb):
        if dai_luong == "delta":
            return sb - sa
        if dai_luong == "delta_rel":
            return (sb - sa) / sa if sa > 0 else float("nan")
        if dai_luong == "R":
            return sb / sa if sa > 0 else float("nan")
        raise ValueError(dai_luong)

    diem = np.nanmean([mot(_ap(a[s], None), _ap(b[s], None)) for s in chung])
    rng = np.random.default_rng(seed)
    mau = np.empty(B)
    for i in range(B):
        ss = rng.choice(chung, size=len(chung), replace=True)
        vals = []
        for s in ss:
            n = len(a[s].img_ids)
            assert n == len(b[s].img_ids), "hai cau hinh phai cung tap test"
            idx = rng.integers(0, n, n)                     # CUNG chi so anh cho ca hai cau hinh
            vals.append(mot(_ap(a[s], idx), _ap(b[s], idx)))
        mau[i] = np.nanmean(vals)
    lo, hi = np.nanpercentile(mau, [2.5, 97.5])
    return float(diem), float(lo), float(hi)


def bootstrap_R_khong_ghep(moc_x, cfg_x, moc_y, cfg_y, B=B_MAC_DINH, seed=0):
    """Hieu R_x - R_y khi hai phia KHONG cung lan lay mau (phep thu T: s5-s9 vs s0-s2).
    Moi phia van ghep cap voi moc A tren chinh lan lay mau cua no; hai phia lay lai doc lap."""
    def r_mau(moc, cfg, rng):
        ss = rng.choice(sorted(set(moc) & set(cfg)), size=len(set(moc) & set(cfg)), replace=True)
        v = []
        for s in ss:
            n = len(moc[s].img_ids)
            idx = rng.integers(0, n, n)
            sa = _ap(moc[s], idx)
            v.append(_ap(cfg[s], idx) / sa if sa > 0 else float("nan"))
        return np.nanmean(v)

    def r_diem(moc, cfg):
        return np.nanmean([_ap(cfg[s], None) / _ap(moc[s], None) for s in sorted(set(moc) & set(cfg))])

    diem = r_diem(moc_x, cfg_x) - r_diem(moc_y, cfg_y)
    rng = np.random.default_rng(seed)
    mau = np.array([r_mau(moc_x, cfg_x, rng) - r_mau(moc_y, cfg_y, rng) for _ in range(B)])
    lo, hi = np.nanpercentile(mau, [2.5, 97.5])
    return float(diem), float(lo), float(hi)


def gop_nhieu_bo(ham_moi_bo, cac_bo, B=B_MAC_DINH, seed=0):
    """Trung binh mot dai luong tren nhieu bo du lieu (cac bo co dinh; lay lai doc lap trong tung bo).
    ham_moi_bo(bo, rng_or_None) -> gia tri (None = uoc luong diem)."""
    diem = float(np.mean([ham_moi_bo(bo, None) for bo in cac_bo]))
    rng = np.random.default_rng(seed)
    mau = np.array([np.mean([ham_moi_bo(bo, rng) for bo in cac_bo]) for _ in range(B)])
    lo, hi = np.nanpercentile(mau, [2.5, 97.5])
    return diem, float(lo), float(hi)


# ----------------------------------------------------------------------------- quy tac (Bang 8)
def ba_trang_thai(diem, lo, hi, delta=DELTA_R):
    """Quy tac ba trang thai doi xung cho moi nhom nguyen nhan (Muc 3.4)."""
    if lo > 0 and diem >= delta:
        return "Duong"
    if hi < delta:
        return "Am"
    return "Khong ro"


def delta_rel_tu_cv(ap_theo_bo):
    """delta_rel = max(2 * CV_trung_binh, 0.05); ap_theo_bo: {bo: [AP cua A qua cac lan lay mau]}."""
    cvs = [np.std(v, ddof=1) / np.mean(v) for v in ap_theo_bo.values()]
    cv_tb = float(np.mean(cvs))
    return max(2 * cv_tb, 0.05), cv_tb, {k: float(c) for k, c in zip(ap_theo_bo, cvs)}


def trang_thai_cai_thien(diem, lo, hi, delta_rel):
    """Trang thai cua mot bo giu lai (Bang 8): 'Cai thien' | 'Kem hon' | 'Khong khac biet ro'."""
    if lo > 0 and diem >= delta_rel:
        return "Cai thien"
    if hi < 0:
        return "Kem hon"
    return "Khong khac biet ro"


def ket_luan_tong_hop(cac_trang_thai):
    """Ket luan tong hop cho mot nhom bo giu lai (Bang 8)."""
    n = len(cac_trang_thai)
    ct = sum(t == "Cai thien" for t in cac_trang_thai)
    kh = sum(t == "Kem hon" for t in cac_trang_thai)
    if ct >= math.ceil(2 * n / 3) and kh == 0:
        return "Cai thien"
    if ct >= 1 and kh == 0:
        return "Cai thien co dieu kien"
    if ct >= 1 and kh >= 1:
        return "Danh doi"
    return "Khong cai thien"


def trang_thai_K(k1, k2, k3):
    """Nhom K Duong neu mot nhanh Duong; Am neu ca ba Am; con lai Khong ro."""
    ds = [k1, k2, k3]
    if "Duong" in ds:
        return "Duong"
    if all(d == "Am" for d in ds):
        return "Am"
    return "Khong ro"


def bang_quyet_dinh(tt, c0_sup_do=True, k1_duong_va_C_khong_sup=False):
    """Bang 9. tt = {'T','D','N','K'} -> trang thai. Tra ve (ma quy tac, ket luan, danh sach bo sung)."""
    if not c0_sup_do:
        return "Q0", "Khong ap dung Bang 9 - kich ban S0 (khong tai lap duoc su sup do)", []
    duong = [k for k in "TDNK" if tt[k] == "Duong"]
    khong_ro = [k for k in "TDNK" if tt[k] == "Khong ro"]
    bo_sung = []
    if "D" in duong and "N" in duong:
        bo_sung.append("Q5a: D va N khong tach duoc hoan toan -> phat bieu theo ti so dung luong/du lieu")
    if k1_duong_va_C_khong_sup:
        bo_sung.append("Q6: phan lon su sup do trong [1] do mo mang de xuat vung")
    if not duong and not khong_ro:
        return "Q1", "Khong nhom nao trong bon nhom giai thich duoc su sup do", bo_sung
    if not duong:
        return "Q2", f"Khong ket luan duoc (Khong ro: {', '.join(khong_ro)})", bo_sung
    if len(duong) == 1 and not khong_ro:
        return "Q3", f"Mot nhom chi phoi: {duong[0]}", bo_sung
    if len(duong) == 1:
        return "Q4", f"{duong[0]} duoc ung ho; chua loai tru {', '.join(khong_ro)}", bo_sung
    return "Q5", f"Nhieu nguyen nhan cung tac dong: {', '.join(duong)}", bo_sung
