"""Va DINOv2: khi khong co xformers, MemEffAttention dung F.scaled_dot_product_attention
thay vi tinh nguyen ma tran attention (N^2 bo nho).

Ly do: anh do phan giai cao (~5.400 token) x lo 4 x 16 dau -> ma tran attention > 7 GB, het bo nho.
Moi truong goc dung xformers.memory_efficient_attention (cung vai tro). Cung phep toan
softmax(q k^T / sqrt(d)) v, cung ti le 1/sqrt(d); chi khac thu tu tinh -> sai khac so hoc nho.
Idempotent.
"""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/lib/dinov2/layers/attention.py")
DAU = "# [CDFSOD-VA] SDPA khi khong co xformers"
src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
else:
    cu = ('        if (not XFORMERS_AVAILABLE) or (not x.is_cuda):\n'
          '            assert attn_bias is None, "xFormers is required for nested tensors usage"\n'
          '            return super().forward(x)\n')
    assert src.count(cu) == 1
    moi = ('        if (not XFORMERS_AVAILABLE) or (not x.is_cuda):\n'
           '            assert attn_bias is None, "xFormers is required for nested tensors usage"\n'
           f'            {DAU}\n'
           '            B, N, C = x.shape\n'
           '            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)\n'
           '            x = torch.nn.functional.scaled_dot_product_attention(\n'
           '                qkv[0], qkv[1], qkv[2], dropout_p=self.attn_drop.p if self.training else 0.0)\n'
           '            x = x.transpose(1, 2).reshape(B, N, C)\n'
           '            return self.proj_drop(self.proj(x))\n')
    src = src.replace(cu, moi)
    if "\nimport torch\n" not in src:
        src = src.replace("from torch import Tensor", "import torch\nfrom torch import Tensor", 1)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va xong")

# Kiem tuong duong so voi nhanh attention thuan tren du lieu ngau nhien
import sys  # noqa: E402
sys.path.insert(0, os.path.expanduser("~/CDFSOD-benchmark"))
import torch  # noqa: E402
from lib.dinov2.layers.attention import Attention, MemEffAttention  # noqa: E402

torch.manual_seed(0)
a = MemEffAttention(1024, num_heads=16, qkv_bias=True).cuda().eval()
x = torch.randn(2, 3249, 1024, device="cuda")
with torch.no_grad():
    for ten, ctx in [("fp32", torch.autocast("cuda", enabled=False)),
                     ("fp16 autocast", torch.autocast("cuda", dtype=torch.float16))]:
        with ctx:
            y_thuan = Attention.forward(a, x).float()
            y_sdpa = a(x).float()
        print(f"{ten}: sai lech lon nhat {(y_thuan - y_sdpa).abs().max().item():.2e}, "
              f"do lon trung binh {y_thuan.abs().mean().item():.2e}")
