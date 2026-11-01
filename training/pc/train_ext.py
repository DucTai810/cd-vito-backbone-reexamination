"""Diem vao huan luyen mo rong cho de tai (khong sua loi detectron2 / CD-ViTO).

Chuc nang (moi tuy chon mac dinh TAT; khi tat het, hanh vi TRUNG voi tools/train_net.py goc
cong tich luy gradient):
  * tich luy gradient: --accum N (lo hieu dung 16 tren 1 card 16 GB)                        [TN1+]
  * mo phan trich dac trung (DINOv2): --mo-pttd {khong, toan_bo, k<N>}                      [TN2+]
      "toan_bo": 24 khoi + lop nhung + chuan hoa cuoi; "k4": 4 khoi cuoi + chuan hoa cuoi
  * toc do hoc rieng: --ti-le-lr-pttd r -> eta_b = r * SOLVER.BASE_LR (eta_h)
  * giam dan theo khoi (LLRD): --llrd h -> khoi i nhan them he so h^(23-i); lop nhung h^24  [TN8, TN13]
  * bo toi uu: --toi-uu {sgd, adamw}                                                           [TN4, TN13]
  * L2-SP: --l2sp a -> them a*(theta - theta0) vao gradient phan trich dac trung, bo phat trong
    so ve 0 cho nhom nay                                                                     [TN13]
  * LoRA: --lora R -> ma tran hang thap tren phep chieu truy van va gia tri cua ca 24 khoi;
    trong so goc giu nguyen                                                                   [TN4]
  * khoa khoi: --khoa-khoi "i,j,..." -> giu nguyen cac khoi nay khi mo toan_bo                [TN9]
  * mo dau mang de xuat vung: --mo-rpn (cau hinh C0)                                          [TN12]
  * che do so hoc DINOv2 khi huan luyen: --so-hoc {fp16, bf16, fp32}; khi CHAM luon fp16      [TN12]
  * gradient checkpointing cho cac khoi can gradient (khong doi ket qua)
  * nhat ky P4 moi buoc (p4.tsv): chuan gradient theo nhom, so gradient khong huu han (cong don)

Chay (trong WSL, env cdfsod):
  python train_ext.py --num-gpus 1 --accum 4 [tuy chon] --config-file <cfg> SOLVER.IMS_PER_BATCH 4 ...
"""
import math
import os
import re
import sys
import time

REPO = os.path.expanduser("~/CDFSOD-benchmark")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.utils.checkpoint as ckpt  # noqa: E402
import train_net as tn  # noqa: E402  (Trainer, setup cua ma goc)
from detectron2.engine import SimpleTrainer, default_argument_parser, launch  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping  # noqa: E402
from detectron2.utils import comm  # noqa: E402
from detectron2.utils.events import get_event_storage  # noqa: E402

# Cau hinh mo rong, dat trong main() tu tham so dong lenh
MO = {"mo_pttd": "khong", "ti_le_lr_pttd": 1.0, "so_hoc": "fp16", "checkpoint": True, "mo_rpn": False,
      "llrd": 1.0, "toi_uu": "sgd", "l2sp": 0.0, "lora": 0, "khoa_khoi": [], "tinh_lai_proto": 0}


# --------------------------------------------------------------------------- LoRA
class LoRAQKV(nn.Module):
    """Boc phep chieu qkv cua DINOv2: them B_q A_q x vao phan truy van va B_v A_v x vao phan gia tri.
    A khoi tao Kaiming, B khoi tao 0 (dau ra ban dau trung ban goc); he so alpha/r = 1.
    Giu DUNG ten 'weight' va 'bias' cua lop goc de khoa trong tep trong so (qkv.weight) van khop
    khi nap sau khi boc (neu dat trong mo-dun con, trong so goc se KHONG duoc nap - loi am tham)."""

    def __init__(self, goc: nn.Linear, r: int):
        super().__init__()
        self.weight = goc.weight
        self.bias = goc.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        d = goc.in_features
        self.C = goc.out_features // 3
        self.A_q = nn.Parameter(torch.empty(r, d)); self.B_q = nn.Parameter(torch.zeros(self.C, r))
        self.A_v = nn.Parameter(torch.empty(r, d)); self.B_v = nn.Parameter(torch.zeros(self.C, r))
        nn.init.kaiming_uniform_(self.A_q, a=math.sqrt(5)); nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))

    def forward(self, x):
        out = nn.functional.linear(x, self.weight, self.bias)
        dq = (x @ self.A_q.t().to(x.dtype)) @ self.B_q.t().to(x.dtype)
        dv = (x @ self.A_v.t().to(x.dtype)) @ self.B_v.t().to(x.dtype)
        C = self.C
        return torch.cat([out[..., :C] + dq, out[..., C:2 * C], out[..., 2 * C:] + dv], dim=-1)


# --------------------------------------------------------------------------- mo phan trich dac trung
def ap_dung_mo_pttd(model):
    """Dat requires_grad theo cau hinh MO; tra ve so tham so cua phan trich dac trung duoc cap nhat."""
    bb = model.backbone
    model._cdfsod_so_hoc = MO["so_hoc"]
    if MO["mo_rpn"]:
        model._cdfsod_mo_rpn = True
        for p in model.offline_proposal_generator.parameters():
            p.requires_grad = True
        print(f"[NHOM THAM SO] mo dau RPN: "
              f"{sum(p.numel() for p in model.offline_proposal_generator.parameters()) / 1e6:.2f} trieu", flush=True)
    n = len(bb.blocks)
    khoi_can_grad = []
    if MO["lora"] > 0:
        assert MO["mo_pttd"] == "khong", "LoRA dung khi khong mo truc tiep phan trich dac trung"
        for blk in bb.blocks:
            blk.attn.qkv = LoRAQKV(blk.attn.qkv, MO["lora"]).to(next(blk.parameters()).device)
        khoi_can_grad = list(range(n))
    elif MO["mo_pttd"] == "toan_bo":
        for p in bb.parameters():
            p.requires_grad = True
        for i in MO["khoa_khoi"]:
            for p in bb.blocks[i].parameters():
                p.requires_grad = False
        khoi_can_grad = [i for i in range(n) if i not in MO["khoa_khoi"]]
    elif re.fullmatch(r"k\d+", MO["mo_pttd"]):     # "k4", "k12"; KHONG khop "khong"
        k = int(MO["mo_pttd"][1:])
        khoi_can_grad = list(range(n - k, n))
        for i in khoi_can_grad:
            for p in bb.blocks[i].parameters():
                p.requires_grad = True
        for p in bb.norm.parameters():            # chuan hoa cuoi ap len dau ra duoc dung
            p.requires_grad = True
    elif MO["mo_pttd"] != "khong":
        raise ValueError(MO["mo_pttd"])
    if not khoi_can_grad:
        return 0
    model._cdfsod_mo_pttd = True
    if MO["checkpoint"]:
        # checkpoint tu khoi can gradient dau tien tro len (cac khoi sau do deu can luu kich hoat)
        dau = min(khoi_can_grad)
        for i in range(dau, n):
            blk = bb.blocks[i]
            goc = blk.forward

            def forward(x, _goc=goc, _blk=blk):
                if torch.is_grad_enabled():
                    return ckpt.checkpoint(_goc, x, use_reentrant=False)
                return _goc(x)
            blk.forward = forward
    return sum(p.numel() for p in bb.parameters() if p.requires_grad)


def ten_nhom(model):
    """id tham so -> ten nhom (G_pttd cho phan trich dac trung, G_con cho phan con lai)."""
    ids_bb = {id(p) for p in model.backbone.parameters()}
    return {id(p): ("G_pttd" if id(p) in ids_bb else "G_con") for p in model.parameters()}


def he_so_llrd(ten_tham_so, n_khoi, h):
    """He so nhan toc do hoc cho tham so phan trich dac trung theo do sau (khoi tren cung = 1)."""
    if h == 1.0:
        return 1.0
    m = re.search(r"blocks\.(\d+)\.", ten_tham_so)
    if m:
        return h ** (n_khoi - 1 - int(m.group(1)))
    if ".norm." in ten_tham_so or ten_tham_so.startswith("norm."):
        return 1.0
    return h ** n_khoi                          # lop nhung, pos_embed, cls_token, mask_token


# --------------------------------------------------------------------------- TN12-K2
class TinhLaiProto:
    """K2: sau moi N buoc, tinh dac trung trung binh cua cac hop anh mau theo tung lop bang phan trich
    dac trung HIEN TAI (ROIAlign 7x7 tren token cua lop dung cho phat hien, chi doi kich thuoc tat dinh),
    roi DICH vector dai dien lop da hoc (class_weights, K dong moi lop) mot luong bang do troi cua dac
    trung lop so voi lan tinh truoc. Giu phan bu da hoc qua gradient.
    Gioi han: vector dai dien nen (bg_tokens) dung tu COCO - de tai khong co COCO nen khong tinh lai."""

    def __init__(self, cfg, model, chu_ky):
        from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
        import detectron2.data.transforms as T
        mapper = DatasetMapper(is_train=True, image_format=cfg.INPUT.FORMAT,
                               augmentations=[T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)])
        self.loader = build_detection_test_loader(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]), mapper=mapper)
        self.model, self.chu_ky, self.truoc = model, chu_ky, None
        self.K = model.k_shot
        self.buoc_anh = 1.0 / model.backbone.patch_size

    @torch.no_grad()
    def dac_trung_lop(self):
        from torchvision.ops import roi_align
        m = self.model
        dang_hl = m.training
        m.eval()
        tong, dem = {}, {}
        for batch in self.loader:
            anh = m.preprocess_image(batch)
            with torch.autocast("cuda", enabled=(m._cdfsod_so_hoc != "fp32"),
                                dtype=torch.bfloat16 if m._cdfsod_so_hoc == "bf16" else torch.float16):
                tok = m.backbone(anh.tensor)[m.vit_feat_name]
            inst = batch[0]["instances"].to(tok.device)
            if len(inst) == 0:
                continue
            f = roi_align(tok.float(), [inst.gt_boxes.tensor], output_size=7,
                          spatial_scale=self.buoc_anh, aligned=True).mean(dim=(2, 3))
            for c, v in zip(inst.gt_classes.tolist(), f):
                tong[c] = tong.get(c, 0) + v
                dem[c] = dem.get(c, 0) + 1
        m.train(dang_hl)
        return {c: tong[c] / dem[c] for c in tong}

    def cap_nhat(self, buoc):
        moi = self.dac_trung_lop()
        if self.truoc is not None:
            W = self.model.class_weights.weight
            do_troi = []
            for c, v in moi.items():
                d = (v - self.truoc[c]).to(W.dtype)
                W.data[c * self.K:(c + 1) * self.K] += d
                do_troi.append(d.norm().item())
            get_event_storage().put_scalar("k2_do_troi_tb", sum(do_troi) / len(do_troi), smoothing_hint=False)
            print(f"[K2] buoc {buoc}: dich vector dai dien, do troi trung binh {sum(do_troi) / len(do_troi):.4f}", flush=True)
        self.truoc = moi


# --------------------------------------------------------------------------- vong huan luyen
class AccumTrainer(SimpleTrainer):
    """SimpleTrainer voi tich luy gradient, L2-SP va nhat ky P4 (chuan gradient theo nhom, gradient khong huu han)."""

    def __init__(self, model, data_loader, optimizer, accum, thu_muc_ra=None):
        super().__init__(model, data_loader, optimizer)
        self.accum = accum
        nhom = ten_nhom(model)
        self._nhom = {"G_pttd": [], "G_con": []}
        for p in model.parameters():
            if p.requires_grad:
                self._nhom[nhom[id(p)]].append(p)
        self._luy_ke = {k: 0 for k in self._nhom}      # so buoc co gradient khong huu han (cong don)
        self._l2sp = []
        if MO["l2sp"] > 0:
            self._l2sp = [(p, p.detach().clone()) for p in self._nhom["G_pttd"]]
        self._tep_p4 = None
        if thu_muc_ra:
            self._tep_p4 = open(os.path.join(thu_muc_ra, "p4.tsv"), "w")
            self._tep_p4.write("buoc\tnhom\tchuan_grad\tso_tham_so_khong_huu_han\n")

    def run_step(self):
        assert self.model.training, "[AccumTrainer] model was changed to eval mode!"
        k2 = getattr(self, "k2", None)
        if k2 is not None and self.iter == 0:
            k2.cap_nhat(0)                       # moc dac trung lop ban dau
        self.optimizer.zero_grad()
        tong_loss = {}
        data_time = 0.0
        for _ in range(self.accum):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time += time.perf_counter() - start
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}
            (sum(loss_dict.values()) / self.accum).backward()
            for k, v in loss_dict.items():
                tong_loss[k] = tong_loss.get(k, 0.0) + v.detach() / self.accum
        self._write_metrics(tong_loss, data_time)
        if self._l2sp:
            with torch.no_grad():
                for p, p0 in self._l2sp:
                    if p.grad is not None:
                        p.grad.add_(p - p0, alpha=MO["l2sp"])
        self._ghi_p4()
        self.optimizer.step()
        if k2 is not None and (self.iter + 1) % k2.chu_ky == 0:
            k2.cap_nhat(self.iter + 1)

    @torch.no_grad()
    def _ghi_p4(self):
        """P4: chuan gradient theo nhom va so tham so co gradient khong huu han (chi ghi, khong chan)."""
        storage = get_event_storage()
        for ten, ps in self._nhom.items():
            gs = [p.grad for p in ps if p.grad is not None]
            if not gs:
                continue
            chuan = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gs]))
            khong_huu_han = sum(int((~torch.isfinite(g)).any()) for g in gs)
            self._luy_ke[ten] += int(khong_huu_han > 0)
            storage.put_scalar(f"p4_chuan_grad_{ten}", chuan.item(), smoothing_hint=False)
            storage.put_scalar(f"p4_khong_huu_han_{ten}", khong_huu_han, smoothing_hint=False)
            storage.put_scalar(f"p4_buoc_khong_huu_han_luy_ke_{ten}", self._luy_ke[ten], smoothing_hint=False)
            if self._tep_p4:
                self._tep_p4.write(f"{self.iter}\t{ten}\t{chuan.item():.6g}\t{khong_huu_han}\n")
                self._tep_p4.flush()
            if self.iter == 0:
                print(f"[P4] buoc 0 {ten}: {len(gs)}/{len(ps)} tham so co gradient, chuan {chuan.item():.3e}, "
                      f"khong huu han {khong_huu_han}", flush=True)


class TrainerExt(tn.Trainer):
    def __init__(self, cfg, accum):
        super().__init__(cfg)
        dung_mo_rong = (accum > 1 or MO["mo_pttd"] != "khong" or MO["lora"] > 0 or MO["mo_rpn"]
                        or MO["l2sp"] > 0 or MO["tinh_lai_proto"] > 0)
        if dung_mo_rong:
            t = self._trainer
            self._trainer = AccumTrainer(t.model, t.data_loader, t.optimizer, accum, cfg.OUTPUT_DIR)
            if MO["tinh_lai_proto"] > 0:
                self._trainer.k2 = TinhLaiProto(cfg, t.model, MO["tinh_lai_proto"])

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        n_mo = ap_dung_mo_pttd(model)
        n_tong = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[NHOM THAM SO] mo_pttd={MO['mo_pttd']} lora={MO['lora']} khoa_khoi={MO['khoa_khoi']}: "
              f"pttd {n_mo / 1e6:.2f} trieu, tong duoc cap nhat {n_tong / 1e6:.2f} trieu, "
              f"so hoc DINOv2={MO['so_hoc']}, toi_uu={MO['toi_uu']}, llrd={MO['llrd']}, l2sp={MO['l2sp']}", flush=True)
        return model

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """Moi luot CHAM bang fp16 nhu ma goc (ke ca luot huan luyen bf16/fp32), de khac biet giua
        cac cau hinh chi den tu khau huan luyen."""
        cu = getattr(model, "_cdfsod_so_hoc", "fp16")
        model._cdfsod_so_hoc = "fp16"
        try:
            return super().test(cfg, model, evaluators)
        finally:
            model._cdfsod_so_hoc = cu

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Nhu detectron2.solver.build_optimizer goc; tham so phan trich dac trung nhan lr * r (* LLRD);
        L2-SP bo phat trong so ve 0 cho nhom nay; AdamW neu --toi-uu adamw."""
        params = get_default_optimizer_params(
            model, base_lr=cfg.SOLVER.BASE_LR, weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR, weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS)
        ten_bb = {id(p): n for n, p in model.backbone.named_parameters()}
        n_khoi = len(model.backbone.blocks)
        for g in params:
            p = g["params"][0]
            if id(p) in ten_bb:
                g["lr"] = g["lr"] * MO["ti_le_lr_pttd"] * he_so_llrd(ten_bb[id(p)], n_khoi, MO["llrd"])
                if MO["l2sp"] > 0:
                    g["weight_decay"] = 0.0
        if MO["toi_uu"] == "adamw":
            return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
            params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


# --------------------------------------------------------------------------- bo nho
def boc_theo_doi_bo_nho(model, chu_ky=20):
    """Boc forward khi suy luan: ghi bo nho da cap phat moi `chu_ky` anh va giai phong bo dem.
    Chi dung de chan doan ro ri / phan manh bo nho; khong doi ket qua du doan."""
    goc = model.forward
    dem = {"n": 0}

    def forward(batched_inputs):
        out = goc(batched_inputs)
        dem["n"] += 1
        if dem["n"] % chu_ky == 0:
            print(f"[BO NHO] anh {dem['n']}: cap phat {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
                  f"giu cho {torch.cuda.memory_reserved() / 2**30:.2f} GiB, "
                  f"dinh {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            torch.cuda.empty_cache()
        return out

    model.forward = forward
    return model


def dat_tran_vram():
    """Tren Windows/WSL, driver am tham tran VRAM sang RAM he thong thay vi bao het bo nho,
    nen bo cap phat PyTorch khong bao gio tu giai phong bo dem -> cham hang tram lan.
    Dat tran de bo cap phat tu giai phong bo dem va thu lai khi cham tran. Khong doi ket qua tinh toan."""
    ti_le = float(os.environ.get("CDFSOD_TRAN_VRAM", "0.88"))
    if torch.cuda.is_available() and ti_le < 1.0:
        torch.cuda.set_per_process_memory_fraction(ti_le)
        print(f"[BO NHO] tran VRAM = {ti_le:.2f} x {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")


# --------------------------------------------------------------------------- main
def main(args):
    MO.update(mo_pttd=args.mo_pttd, ti_le_lr_pttd=args.ti_le_lr_pttd, so_hoc=args.so_hoc,
              checkpoint=not args.khong_checkpoint, mo_rpn=args.mo_rpn, llrd=args.llrd, toi_uu=args.toi_uu,
              l2sp=args.l2sp, lora=args.lora,
              khoa_khoi=[int(x) for x in args.khoa_khoi.split(",") if x.strip() != ""],
              tinh_lai_proto=args.tinh_lai_proto)
    dat_tran_vram()
    cfg = tn.setup(args)
    if args.eval_only:
        model = tn.Trainer.build_model(cfg)
        if MO["lora"] > 0:                      # can dung lai cau truc LoRA de nap trong so
            ap_dung_mo_pttd(model)
        model._cdfsod_so_hoc = "fp16"          # cham luon bang fp16 nhu ma goc
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        if os.environ.get("CDFSOD_THEO_DOI_BO_NHO"):
            boc_theo_doi_bo_nho(model)
        return tn.Trainer.test(cfg, model)
    trainer = TrainerExt(cfg, args.accum)
    trainer.resume_or_load(resume=args.resume)
    ket_qua = trainer.train()
    if comm.is_main_process():
        print("BO NHO DINH (GiB):", round(torch.cuda.max_memory_allocated() / 2**30, 2),
              "| giu cho:", round(torch.cuda.max_memory_reserved() / 2**30, 2))
    return ket_qua


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--accum", type=int, default=1, help="so lo nho tich luy truoc moi buoc cap nhat")
    parser.add_argument("--mo-pttd", default="khong", help="khong | toan_bo | k<N> (N khoi cuoi)")
    parser.add_argument("--ti-le-lr-pttd", type=float, default=1.0, help="eta_b / eta_h")
    parser.add_argument("--so-hoc", default="fp16", choices=["fp16", "bf16", "fp32"], help="che do so hoc DINOv2")
    parser.add_argument("--khong-checkpoint", action="store_true", help="tat gradient checkpointing")
    parser.add_argument("--mo-rpn", action="store_true", help="mo dau mang de xuat vung (cau hinh C0)")
    parser.add_argument("--llrd", type=float, default=1.0, help="he so giam toc do hoc theo khoi")
    parser.add_argument("--toi-uu", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--l2sp", type=float, default=0.0, help="cuong do L2-SP")
    parser.add_argument("--lora", type=int, default=0, help="hang LoRA (0 = tat)")
    parser.add_argument("--khoa-khoi", default="", help="danh sach khoi giu nguyen, vd '3,7,12'")
    parser.add_argument("--tinh-lai-proto", type=int, default=0, help="K2: dich vector dai dien moi N buoc (0 = tat)")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
