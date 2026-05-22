# Malicious Websites Detection Agent
是一個整合多模態、A2A、MCP的模型框架。
當中使用到三種模態分別為 URL序列化特徵、HTML(DOM)圖結構、網站伺服器行為統計型特徵。採取中間層融合的方式將三個子模型做為編碼器，使用各自的嵌入向量進行映射到統一的特徵維度進行特徵對齊，之後輸入進兩層全連接網路進行特徵融合。這個過程中，各個模型將被打包做為Agent，調用這些Agent的任務交由Llama3進行決策。
# Framework
![MWDA框架](/系統應用框架.png)
# 使用範例—以Claude Desktop做為使用介面
![1](/1.PNG)
![3](/3.PNG)
![4](/4.PNG)
![2](/2.PNG)