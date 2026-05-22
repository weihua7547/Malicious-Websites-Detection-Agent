import pickle
import torch
import numpy as np

class Response_lightGBM_Agent:
    def __init__(self, lgb_path, device):
        self.device = device
        with open(lgb_path, "rb") as f:
            self.lgb_model = pickle.load(f)["model"]

    def extract(self, rpc_request):
        """
        Input (JSON-RPC Request):
        { "params": {"features": np.array}, "id": "..." }
        """
        params = rpc_request.get("params", {})
        lgb_features = params.get("features")
        req_id = rpc_request.get("id", "unknown")

        leaf_indices = self.lgb_model.predict(lgb_features, pred_leaf=True)
        leaf_tensor = torch.tensor(leaf_indices, dtype=torch.long).to(self.device)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "leaf_indices": leaf_tensor,
                "num_trees": leaf_tensor.shape[1],
                "agent": "StatisticsPerceptionAgent"
            }
        }