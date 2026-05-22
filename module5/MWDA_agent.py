# =========================================================
# MWDA Live URL Inference API
# =========================================================

from murl_bert_agent import MURL_BERT_Agent
from rac_tagcn_agent import RAC_TAGCN_Agent
from response_lightGBM_agent import Response_lightGBM_Agent
from fusion_agent import FusionAgent

from preprocessor import html_to_graph_data

import torch
import requests
import json
import pickle
import os
import re
import numpy as np
import faiss
import socket
import math
import ssl
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import tldextract

# =========================================================
# Global Config
# =========================================================
BERT_PATH = "module2/BERT/finetune/DAPT_TAPT/bert_url_best"

RAC_PATH = (
    "module3/db_data/checkpoints_rac/"
    "DAPT/TOP_K_3/rac_best.pt"
)

LIGHTGBM_PATH = (
    "module2/LightGBM/checkpoints/"
    "lightgbm_best.pkl"
)

FUSION_CKPT = (
    "module4/bert_rac_gbm/best_fusion.pt"
)

MODALITIES = "bert,rac,leaf"

HIDDEN_DIM = 64

MLP_HIDDEN_DIM = 512

OLLAMA_MODEL = "llama3.1:8b"

TOP_K = 50

SIM_THRESHOLD = 0.65

JSONL_ROOT = "module1/data_preprocessing/output/finetune/"

FAISS_INDEX_PATH = "module3/db_data/html.faiss"

ID_MAPPING_PATH = "module3/db_data/id_mapping.pkl"


# =========================================================
# Label Map
# =========================================================
LABEL_MAP_INV = {
    0: "benign",
    1: "phishing",
    2: "defacement",
    3: "malware"
}


# =========================================================
# Load Retrieval Models
# =========================================================
# print("🔹 Loading SentenceTransformer...")

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# print("🔹 Loading FAISS...")

faiss_index = faiss.read_index(
    FAISS_INDEX_PATH
)

# print("🔹 Loading ID Mapping...")

with open(ID_MAPPING_PATH, "rb") as f:
    id_mapping = pickle.load(f)

# print(f"✅ FAISS vectors: {faiss_index.ntotal}")


# =========================================================
# HTML Normalize
# =========================================================
def normalize_html(html):

    if not html:
        return ""

    html = re.sub(
        r"<script.*?>.*?</script>",
        "[SCRIPT]",
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    html = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    html = re.sub(
        r'(?:[A-Za-z0-9+/]{40,}={0,2})',
        '[Base64]',
        html
    )

    html = re.sub(
        r'(https?://[^\s"\'>]+|www\.[^\s"\'>]+)',
        '[URL]',
        html,
        flags=re.IGNORECASE
    )

    html = re.sub(
        r'\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64}|[a-fA-F0-9]{128})\b',
        '[Hash]',
        html
    )

    html = re.sub(
        r'\b\d{6,}\b',
        '[NUM]',
        html
    )

    html = re.sub(
        r'[\t\r\n]+',
        ' ',
        html
    )

    html = re.sub(
        r'\s+',
        ' ',
        html
    )

    return html.strip()


# =========================================================
# Load HTML from JSONL
# =========================================================
def load_html_from_jsonl(file_name, line_idx):

    path = os.path.join(
        JSONL_ROOT,
        file_name
    )

    with open(path, "r", encoding="utf-8") as f:

        for i, line in enumerate(f):

            if i == line_idx:

                obj = json.loads(line)

                return obj.get("html", "")

    return ""


# =========================================================
# Search Neighbor HTML
# =========================================================
def search_neighbors(query_html):

    struct_text = query_html

    query_vec = embed_model.encode(
        [struct_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = faiss_index.search(
        query_vec,
        TOP_K
    )

    results = []

    for idx, score in zip(indices[0], distances[0]):

        if idx >= len(id_mapping):
            continue

        if score < SIM_THRESHOLD:
            continue

        file_name, line_idx = id_mapping[idx]

        html = load_html_from_jsonl(
            file_name,
            line_idx
        )

        if not html:
            continue

        html = normalize_html(html)

        results.append({
            "score": float(score),
            "html": html
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:3]
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)
def get_ip(hostname: str) -> str:
    if not hostname or len(hostname.strip()) == 0 or len(hostname) > 253:
        return ""
    try:
        return socket.gethostbyname(hostname)
    except Exception:
        return ""
def fetch_ssl_cert(hostname: str, port: int = 443, timeout: float = 4.0):
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            return ssock.getpeercert()
def cert_is_self_signed(cert: dict) -> bool:
    try:
        return cert["issuer"] == cert["subject"]
    except KeyError:
        return False

def cert_days_left(cert: dict) -> int:
    try:
        expire_ts = ssl.cert_time_to_seconds(cert["notAfter"])
        return int((expire_ts - time.time()) / 86400)
    except Exception:
        return -9999
def get_asn_info(ip: str) -> str:
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}?fields=as", timeout=4)
        if r.status_code == 200:
            raw_as = r.json().get("as", "")
            # 使用 \d+ 尋找字串中的所有數字並串接起來
            asn_digits = "".join(re.findall(r'\d+', raw_as))
            return asn_digits
    except:
        return ""
    return ""
SUSPICIOUS_JS_REGEX = re.compile(r"\b(eval|atob|unescape|fromCharCode)\b", re.I)
# =========================================================
# Response Feature Extraction
# =========================================================
def extract_response_features(url, resp, html):
    
    headers = resp.headers
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    hostname = parsed.hostname
    if not hostname or len(hostname) > 253:
        return None  # hostname 空或太長，跳過

    ip = get_ip(hostname)
    if not ip:
        return None  # 無法取得 IP，跳過

    asn = get_asn_info(ip)

    tls_self_signed = None
    tls_days_left = None
    if parsed.scheme == "https" and hostname:
        try:
            cert = fetch_ssl_cert(hostname, port)
            if cert:
                tls_self_signed = int(cert_is_self_signed(cert))
                tls_days_left = cert_days_left(cert)
        except:
            pass
    orig_domain = tldextract.extract(url).registered_domain
    final_domain = tldextract.extract(resp.url).registered_domain
    redirect_external = int(orig_domain != final_domain)
    hdrs = {k.lower(): v for k, v in resp.headers.items()}
    server_entropy = shannon_entropy(hdrs.get("server", ""))

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    server = headers.get(
        "Server",
        ""
    )

    features = {}

    features["asn"] = asn

    features["tls_self_signed"] = tls_self_signed

    features["tls_days_left"] = tls_days_left

    features["http_code"] = resp.status_code

    features["is_redirect"] = int(
        len(resp.history) > 0
    )

    features["redirect_external"] = redirect_external

    features["date_header_present"] = int(
        "Date" in headers
    )

    features["server_entropy"] = server_entropy

    features["missing_csp"] = int(
        "Content-Security-Policy"
        not in headers
    )

    features["html_length"] = len(html)

    features["content_length_hdr"] = int(
        headers.get(
            "Content-Length",
            0
        ) or 0
    )

    features["suspicious_js_cnt"] = len(SUSPICIOUS_JS_REGEX.findall(resp.text))

    features["has_iframe"] = int(
        len(soup.find_all("iframe")) > 0
    )

    return features


# =========================================================
# Build Sample
# =========================================================
def build_sample_from_url(url):

    # print(f"\n🌐 Fetching: {url}")

    resp = requests.get(
        url,
        timeout=10,
        allow_redirects=True,
        headers={
            "User-Agent":
            "Mozilla/5.0"
        }
    )

    raw_html = resp.text

    html = normalize_html(raw_html)

    # print("✅ HTML normalized")

    retrieval_results = search_neighbors(
        html
    )

    neighbor_htmls = [
        r["html"]
        for r in retrieval_results
    ]

    # print(
    #     f"✅ Retrieved "
    #     f"{len(neighbor_htmls)} neighbors"
    # )

    q_graph = html_to_graph_data(html)

    neighbors = []

    for h in neighbor_htmls:

        try:

            g = html_to_graph_data(h)

            if g is not None:
                neighbors.append(g)

        except Exception as e:
            continue
            # print(
            #     f"Neighbor graph error: {e}"
            # )

    if len(neighbors) == 0:
        neighbors = [q_graph] * 3

    feature_dict = extract_response_features(
        url,
        resp,
        html
    )

    lgb_features = np.array(
        [list(feature_dict.values())],
        dtype=np.float32
    )

    return {
        "url": url,
        "html": html,
        "q_graph": q_graph,
        "neighbors": neighbors,
        "lgb_features": lgb_features
    }


# =========================================================
# Feature Dict
# =========================================================
def build_feature_dict(raw_sample):

    f = raw_sample["lgb_features"][0]

    return {
        "asn": int(f[0]),
        "tls_self_signed": int(f[1]),
        "tls_days_left": int(f[2]),
        "http_code": int(f[3]),
        "is_redirect": int(f[4]),
        "redirect_external": int(f[5]),
        "date_header_present": int(f[6]),
        "server_entropy": float(f[7]),
        "missing_csp": int(f[8]),
        "html_length": int(f[9]),
        "content_length_hdr": int(f[10]),
        "suspicious_js_cnt": int(f[11]),
        "has_iframe": int(f[12]),
    }

def stats_to_text(features_dict):
    return f"""
    [Network & Response Features]
    - ASN: {features_dict['asn']}
    - TLS self-signed: {features_dict['tls_self_signed']} (1=yes, 0=no)
    - TLS days remaining: {features_dict['tls_days_left']}
    - HTTP status code: {features_dict['http_code']}
    - Redirect: {features_dict['is_redirect']} (1=yes, 0=no)
    - External redirect: {features_dict['redirect_external']} (1=yes, 0=no)
    - Date header present: {features_dict['date_header_present']}
    - Server entropy: {features_dict['server_entropy']:.3f}
    - Missing CSP: {features_dict['missing_csp']}
    - HTML length: {features_dict['html_length']}
    - Content-Length header: {features_dict['content_length_hdr']}
    - Suspicious JS count: {features_dict['suspicious_js_cnt']}
    - Contains iframe: {features_dict['has_iframe']}
    """
# =========================================================
# Ollama Coordinator
# =========================================================
class OllamaCoordinator:
    def __init__(self, model_name="llama3.1:8b"):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def dispatch(self, url, html_summary, features_dict):
        stats_text = stats_to_text(features_dict)
        """
        透過本地 Ollama 模型進行邏輯派發
        """
        # 針對決策Llama的prompt
        prompt = f"""
        [System: Anti-Malicious Website Orchestrator]
        Analyze the URL and HTML summary to decide which agents to activate.please use all agents.
        URL: {url}
        HTML Summary: {html_summary}
        Response : {stats_text}
        Specialized Agents:
        1. "murl_bert_agent": For lexical/semantic URL patterns.
        2. "rac_tagcn_agent": For DOM tree/visual structure.
        3. "response_lightgbm_agent": For TLS/ASN/HTTP metadata.

        Output MUST be strict JSON format:
        {{"reasoning": "short explanation", "activate": ["agent_name1", "agent_name2"]}}
        """
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"  # 強制 Ollama 輸出 JSON
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status() # 建議增加檢查 HTTP 狀態
            raw_res = response.json().get("response", "{}")
            decision = json.loads(raw_res)
            
            # 確保 activate 是 list，reasoning 是 str
            return decision.get("activate", []), decision.get("reasoning", "")
            
        except Exception as e:
            # 發生錯誤時，回傳空清單與錯誤訊息，確保依然是 2 個回傳值
            error_msg = f"Ollama Error: {str(e)}"
            # print(f"{error_msg}")
            # 預設啟動所有 Agent 以防萬一，或回傳空清單
            default_agents = ["murl_bert_agent", "rac_tagcn_agent", "response_lightgbm_agent"]
            return default_agents, error_msg


# =========================================================
# MWDA
# =========================================================
class MWDA:

    def __init__(self, in_dim):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.brain = OllamaCoordinator()

        self.url_agent = MURL_BERT_Agent(
            BERT_PATH,
            self.device
        )

        from model_arch import (
            TagcnEncoder,
            RACGraphClassifier,
            TriModalMLPFusion
        )

        encoder = TagcnEncoder(
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM
        )

        rac_model = RACGraphClassifier(
            encoder,
            HIDDEN_DIM,
            num_classes=4
        )

        self.struct_agent = RAC_TAGCN_Agent(
            rac_model,
            RAC_PATH,
            self.device
        )

        self.stats_agent = (
            Response_lightGBM_Agent(
                LIGHTGBM_PATH,
                self.device
            )
        )

        # =============================================
        # Fusion
        # =============================================
        with open(
            LIGHTGBM_PATH,
            "rb"
        ) as f:

            lgb_model = pickle.load(f)["model"]

        num_trees = (
            lgb_model.booster_.num_trees()
        )

        num_leaves = (
            lgb_model.get_params()["num_leaves"]
        )

        fusion_model = TriModalMLPFusion(
            modalities=MODALITIES.split(","),
            rac_dim=HIDDEN_DIM * 2,
            mlp_hidden_dim=MLP_HIDDEN_DIM,
            num_trees=num_trees,
            num_leaves=num_leaves
        )

        self.fusioner = FusionAgent(
            fusion_model,
            FUSION_CKPT,
            self.device
        )

    def predict(self, sample):

        features_dict = build_feature_dict(
            sample
        )

        # =============================================
        # Ollama Coordinator
        # =============================================
        active_list, reasoning = (
            self.brain.dispatch(
                sample["url"],
                sample["html"],
                features_dict
            )
        )

        # print("\n🧠 Ollama Decision")
        # print("Activated:", active_list)
        # print("Reasoning :", reasoning)

        # =============================================
        # Default Zero Embeddings
        # =============================================

        bert_emb = torch.zeros(
            (1, 768),
            dtype=torch.float32,
            device=self.device
        )

        rac_emb = torch.zeros(
            (1, HIDDEN_DIM * 2),
            dtype=torch.float32,
            device=self.device
        )

        # leaf idx shape:
        # [batch, num_trees]
        with open(LIGHTGBM_PATH, "rb") as f:
            lgb_model = pickle.load(f)["model"]

        num_trees = (
            lgb_model.booster_.num_trees()
        )

        leaf_idx = torch.zeros(
            (1, num_trees),
            dtype=torch.long,
            device=self.device
        )

        # =============================================
        # URL Agent
        # =============================================
        if "murl_bert_agent" in active_list:

            resp = self.url_agent.extract({
                "jsonrpc":"2.0",
                "id":"url",
                "params":{
                    "url":sample["url"]
                }
            })

            bert_emb = resp["result"]["embedding"]

        # =============================================
        # RAC Agent
        # =============================================
        if "rac_tagcn_agent" in active_list:

            resp = self.struct_agent.extract({
                "jsonrpc":"2.0",
                "id":"rac",
                "params":{
                    "q_graph":sample["q_graph"],
                    "neighbors":sample["neighbors"]
                }
            })

            rac_emb = resp["result"]["embedding"]

        # =============================================
        # Response Agent
        # =============================================
        if "response_lightgbm_agent" in active_list:

            resp = self.stats_agent.extract({
                "jsonrpc":"2.0",
                "id":"resp",
                "params":{
                    "features":
                    sample["lgb_features"]
                }
            })

            leaf_idx = (
                resp["result"]["leaf_indices"]
            )

        # =============================================
        # Fusion
        # =============================================
        pred_class, probs = (
            self.fusioner.fusion(
                bert_emb,
                rac_emb,
                leaf_idx
            )
        )

        pred_idx = (
            pred_class
            .cpu()
            .numpy()[0]
        )

        pred_label = LABEL_MAP_INV[
            pred_idx
        ]

        confidence = (
            probs
            .cpu()
            .numpy()
            .max()
        )

        return {
            "url": sample["url"],
            "pred_id": int(pred_idx),
            "pred_label": pred_label,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "activated_agents": active_list
        }

# =========================================================
# Global MWDA Singleton
# =========================================================
MWDA_system = None


# =========================================================
# Public API
# =========================================================
def predict_url(url):

    global MWDA_system

    sample = build_sample_from_url(
        url
    )

    if MWDA_system is None:

        in_dim = sample[
            "q_graph"
        ].x.size(1)

        MWDA_system = MWDA(
            in_dim
        )

    result = MWDA_system.predict(
        sample
    )

    return result


# =========================================================
# Example
# =========================================================
if __name__ == "__main__":

    result = predict_url(
        "https://ecare.nfu.edu.tw/"
    )

    print("\n" + "=" * 60)

    print("🚀 MWDA RESULT")

    print("=" * 60)

    print(result)