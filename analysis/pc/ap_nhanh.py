"""Tinh AP kieu COCO nhanh, cho phep lay lai mau theo anh (bootstrap) ma khong chay lai COCOeval.

Y tuong: chay COCOeval.evaluate() MOT lan de co ket qua khop du doan - hop that cho tung
(anh, lop) o 10 nguong IoU (vung 'all', toi da 100 du doan/anh). Sau do, voi mot da tap anh bat ky
(co lap lai), AP duoc tinh bang cach gop cac mang da khop - dung logic COCOeval.accumulate()
(101 diem recall, noi suy precision don dieu).

Kiem: ap_toan_bo() phai trung AP cua COCOeval tren toan tap (xem kiem_thu.py).
"""
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RECALL_THRS = np.linspace(0.0, 1.00, 101)


class BangKhop:
    """Ket qua khop cua mot luot chay tren mot tap test."""

    def __init__(self, gt_json, dt_json):
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(gt_json)
            dts = json.load(open(dt_json)) if isinstance(dt_json, str) else dt_json
            dt = gt.loadRes(dts) if len(dts) else COCO()
            ev = COCOeval(gt, dt, "bbox")
            ev.params.areaRng = [ev.params.areaRng[0]]      # chi vung 'all'
            ev.params.areaRngLbl = ["all"]
            ev.evaluate()
        self.img_ids = list(ev.params.imgIds)
        self.cat_ids = list(ev.params.catIds)
        self.n_iou = len(ev.params.iouThrs)
        max_det = ev.params.maxDets[-1]
        n_img = len(self.img_ids)
        # moi (lop, anh): diem, khop (T x D), bo qua (T x D), so gt khong bo qua
        self.du_lieu = {}
        for k, _ in enumerate(self.cat_ids):
            for i in range(n_img):
                e = ev.evalImgs[k * n_img + i]
                if e is None:
                    continue
                diem = np.array(e["dtScores"][:max_det], dtype=np.float64)
                thu_tu = np.argsort(-diem, kind="mergesort")
                self.du_lieu[(k, i)] = (
                    diem[thu_tu],
                    np.asarray(e["dtMatches"])[:, :max_det][:, thu_tu] > 0,
                    np.asarray(e["dtIgnore"])[:, :max_det][:, thu_tu].astype(bool),
                    int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)),
                )

    def ap(self, chi_so_anh=None):
        """AP (0..100) va AP50 tren da tap anh `chi_so_anh` (mang chi so, co the lap); None = toan bo."""
        if chi_so_anh is None:
            chi_so_anh = np.arange(len(self.img_ids))
        dem = np.bincount(chi_so_anh, minlength=len(self.img_ids))
        ap_lop = []
        for k in range(len(self.cat_ids)):
            diem, khop, bo, npig = [], [], [], 0
            for i in np.nonzero(dem)[0]:
                d = self.du_lieu.get((k, i))
                if d is None:
                    continue
                for _ in range(dem[i]):
                    diem.append(d[0]); khop.append(d[1]); bo.append(d[2]); npig += d[3]
            if npig == 0:
                continue                                     # nhu COCOeval: lop khong co gt -> bo
            if diem:
                diem = np.concatenate(diem)
                tt = np.argsort(-diem, kind="mergesort")
                khop = np.concatenate(khop, axis=1)[:, tt]
                bo = np.concatenate(bo, axis=1)[:, tt]
            else:
                khop = np.zeros((self.n_iou, 0), bool); bo = khop.copy()
            tp = np.cumsum(khop & ~bo, axis=1, dtype=np.float64)
            fp = np.cumsum(~khop & ~bo, axis=1, dtype=np.float64)
            p_t = np.zeros((self.n_iou, len(RECALL_THRS)))
            for t in range(self.n_iou):
                if tp.shape[1] == 0:
                    continue
                rc = tp[t] / npig
                pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                idx = np.searchsorted(rc, RECALL_THRS, side="left")
                hop_le = idx < len(pr)
                p_t[t, hop_le] = pr[idx[hop_le]]
            ap_lop.append(p_t)
        if not ap_lop:
            return float("nan"), float("nan")
        p = np.stack(ap_lop)                                 # lop x T x R
        return 100 * p.mean(), 100 * p[:, 0, :].mean()

    def ap_toan_bo(self):
        return self.ap(None)
