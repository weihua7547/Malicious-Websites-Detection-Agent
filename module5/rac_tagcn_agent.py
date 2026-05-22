import torch
from torch_geometric.data import Batch as PyGBatch

class RAC_TAGCN_Agent:
    def __init__(self, model_instance, state_dict_path, device):
        self.device = device
        self.model = model_instance.to(device)
        self.model.load_state_dict(torch.load(state_dict_path, map_location=device), strict=False)
        self.model.eval()

    def extract(self, rpc_request):
        """
        Input (JSON-RPC Request):
        { "params": {"q_graph": Data, "neighbors": [Data...]}, "id": "..." }
        """
        params = rpc_request.get("params", {})
        q_graph = params.get("q_graph")
        neighbors = params.get("neighbors", [])
        req_id = rpc_request.get("id", "unknown")

        batch_sample = [{"q_graph": q_graph, "neighbors": neighbors, "labels": torch.tensor(0)}]
        with torch.no_grad():
            emb = self.model.get_embedding(batch_sample, self.device)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "embedding": emb,
                "dim": emb.shape[-1],
                "agent": "StructurePerceptionAgent"
            }
        }