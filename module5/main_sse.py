import sys
import logging
import asyncio
import json
from mcp.server.fastmcp import FastMCP   
from classifier_bert import BERT_Detection
from MWDA_agent import predict_url

logging.basicConfig(
    level=logging.DEBUG,      
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MWDA")

# 建立 FastMCP
mcp = FastMCP("MWDA")

@mcp.tool(
    name="MWDA_Detection",
    description="分析輸入的 URL 並回傳其惡意分類類別（如 phishing, malware, defacement 等）。"
)
def MWDA_detection(url: str) -> str:
    logger.info(f"[TOOL CALLED] 收到 URL: {url}")
    result = predict_url(url)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("🚀 Starting MWDA SSE Server -")
    logger.info("工具名稱：MWDA_Detection （應該要出現在 Inspector 中）")
    
    mcp.run(transport="sse")
    