import torch
import torch.nn as nn

class FusionAgent(nn.Module):
    def __init__(self, fusion_model_instance, ckpt_path, device):
        super().__init__()

        self.device = device

        self.fusion_layer = fusion_model_instance.to(device)

        print(f"[Fusion] Loading checkpoint: {ckpt_path}")

        state_dict = torch.load(
            ckpt_path,
            map_location=device
        )

        missing, unexpected = self.fusion_layer.load_state_dict(
            state_dict,
            strict=False
        )

        print(f"[Fusion] Missing keys: {missing}")
        print(f"[Fusion] Unexpected keys: {unexpected}")

        self.fusion_layer.eval()

    def fusion(self, bert_emb, rac_emb, leaf_indices):

        with torch.no_grad():

            # tensor 化
            if bert_emb is not None and not torch.is_tensor(bert_emb):
                bert_emb = torch.tensor(
                    bert_emb,
                    dtype=torch.float32,
                    device=self.device
                )

            if rac_emb is not None and not torch.is_tensor(rac_emb):
                rac_emb = torch.tensor(
                    rac_emb,
                    dtype=torch.float32,
                    device=self.device
                )

            if leaf_indices is not None and not torch.is_tensor(leaf_indices):
                leaf_indices = torch.tensor(
                    leaf_indices,
                    dtype=torch.long,
                    device=self.device
                )

            # batch 維度保護
            if bert_emb is not None and bert_emb.dim() == 1:
                bert_emb = bert_emb.unsqueeze(0)

            if rac_emb is not None and rac_emb.dim() == 1:
                rac_emb = rac_emb.unsqueeze(0)

            if leaf_indices is not None and leaf_indices.dim() == 1:
                leaf_indices = leaf_indices.unsqueeze(0)

            logits = self.fusion_layer(
                bert_emb=bert_emb,
                rac_emb=rac_emb,
                leaf_indices=leaf_indices
            )

            probs = torch.softmax(logits, dim=-1)

            pred_class = torch.argmax(probs, dim=-1)

        return pred_class, probs