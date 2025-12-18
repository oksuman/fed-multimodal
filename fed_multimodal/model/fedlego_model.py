# fed-multimodal/fed_multimodal/experiment/ptb-xl/fedlego_model.py

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# mm-lego README Usage에 나온 공식 import 경로/클래스명
# from mm_lego.models import LegoBlock, MILAttentionNet, SNN, LegoMerge, LegoFuse
from mm_lego.models import LegoBlock, LegoMerge


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


class FedLegoClassifier(nn.Module):
    """
    FedLego = LegoBlock(s) + LegoMerge + task head

    - inputs: list of modality tensors, each typically shaped (B, C, D)
      (PTB-XL은 lead-group을 modality로 두고 (B, 6, T) 형태가 흔함)
    - missing modality: None이면 0-tensor로 대체(가장 보수적이고 구현 쉬움)
    - encoder freeze: 블록 내부 encoder는 freeze 가능
    """

    def __init__(
        self,
        in_shapes: Sequence[Tuple[int, int]],
        num_classes: int,
        encoders: Optional[Sequence[nn.Module]] = None,
        head_method: str = "slerp",
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()

        if encoders is None:
            encoders = [nn.Identity() for _ in in_shapes]

        assert len(encoders) == len(in_shapes), "encoders와 in_shapes 길이가 같아야 합니다."

        self.in_shapes: List[Tuple[int, int]] = list(in_shapes)

        # 1) modality별 LegoBlock 구성
        self.blocks = nn.ModuleList(
            [LegoBlock(in_shape=shp, encoder=enc) for shp, enc in zip(self.in_shapes, encoders)]
        )

        # 2) LegoMerge로 멀티모달 결합
        # README 예시: LegoMerge(blocks=[...], head_method="slerp", final_head=False)
        self.merger = LegoMerge(blocks=list(self.blocks), head_method=head_method, final_head=False)

        # 3) downstream classifier (입력 dim을 몰라도 되도록 LazyLinear)
        self.classifier = nn.LazyLinear(num_classes)

        # 4) encoder freeze
        if freeze_encoders:
            for blk in self.blocks:
                # LegoBlock 안에서 encoder 속성명이 "encoder"라고 가정 (README에서 encoder=로 주입)
                # 만약 로컬 코드에서 이름이 다르면 여기만 맞춰주면 됨.
                if hasattr(blk, "encoder") and isinstance(blk.encoder, nn.Module):
                    freeze_module(blk.encoder)

    def forward(
        self,
        xs: Sequence[Optional[torch.Tensor]],
        return_embeddings: bool = False,
    ):
        # xs 길이 보정
        assert len(xs) == len(self.in_shapes), f"입력 modality 개수({len(xs)}) != in_shapes({len(self.in_shapes)})"

        # missing modality를 0으로 대체
        x_list: List[torch.Tensor] = []
        for i, x in enumerate(xs):
            if x is None:
                # batch size는 다른 modality에서 가져오는 게 안전
                # 모두 None이면 에러 내는 게 맞음
                b = None
                for y in xs:
                    if y is not None:
                        b = y.shape[0]
                        device = y.device
                        dtype = y.dtype
                        break
                if b is None:
                    raise ValueError("All modalities are None (all missing). Cannot run forward.")
                c, d = self.in_shapes[i]
                x = torch.zeros((b, c, d), device=device, dtype=dtype)
            x_list.append(x)

        # mm-lego README: merged_model([tab_data, img_data], return_embeddings=True)
        emb = self.merger(x_list, return_embeddings=True)

        # classifier는 (B, *) 형태를 기대 → flatten
        emb_flat = emb.flatten(1)
        logits = self.classifier(emb_flat)

        if return_embeddings:
            return logits, emb
        return logits


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0):
    # requires_grad=True 인 파라미터만 업데이트 (encoder freeze 정책과 자연스럽게 연동됨)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
