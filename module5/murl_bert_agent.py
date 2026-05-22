import torch
from transformers import BertTokenizer, BertForSequenceClassification

class MURL_BERT_Agent:
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # 載入微調過的 BERT
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()

    def extract(self, rpc_request):
        """
        Input (JSON-RPC Request): 
        { "jsonrpc": "2.0", "method": "extract", "params": {"url": "...", "id": "..."} }
        """
        params = rpc_request.get("params", {})
        url = params.get("url", "")
        req_id = rpc_request.get("id", "unknown")

        inputs = self.tokenizer(url, max_length=256, padding="max_length", 
                                truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.bert(**inputs).pooler_output

        # 返回 JSON-RPC 2.0 Response 格式
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "embedding": emb, # 保持 Tensor 格式供 Coordinator 使用
                "dim": emb.shape[-1],
                "agent": "URLPerceptionAgent"
            }
        }