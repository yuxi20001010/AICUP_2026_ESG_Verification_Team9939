# AI CUP 2026 春季賽 - ESG 永續承諾驗證競賽 TEAM_9939

本儲存庫為 TEAM_9939 參與 [AI CUP 2026 春季賽：ESG 永續承諾驗證競賽] 的**訓練模型** 程式碼。

> **⚠️ 專案定位說明**
> 本儲存庫的程式碼，主要目的為針對 ESG 報告文字訓練四種不同的中文預訓練骨幹模型，並產出 Out-Of-Fold (OOF) 與測試集的預測機率矩陣（`.pkl` 檔）。
> 這些產出的機率特徵，後續會交接給其他團隊成員負責的模型，融入訓練流程中進行最終的權重融合與邏輯約束解碼，從而得出我們團隊的最終提交成績（Private Leaderboard: 0.6375782, Rank 21）。

## 🌟 本階段模型特色與創新性

在作為基礎模型（Base Models）的訓練過程中，本專案針對單一模型引入了以下客製化改良，確保提供給第二階段的機率特徵具有足夠的泛化能力與多樣性：

* **多任務聯合學習架構**: 將四項分類任務（promise_status, verification_timeline, evidence_status, evidence_quality）整合於單一模型中同步學習，使模型學習到跨任務的共同語義表示。
* **Masked Mean Pooling**: 捨棄標準的 `[CLS]` token，對所有有效 token 的隱藏向量依 attention mask 加權平均，使句子表示能均勻涵蓋全文語意，有效減少長句資訊遺失。
* **截斷式類別加權損失**: 針對樣本極度不均的類別（如 evidence_quality 的 Misleading），在計算平衡權重時引入上限截斷機制（`CLASS_WEIGHT_MAX=5.0`），防止稀有類別梯度爆炸。
* **結合 FGM 對抗訓練**: 於共用骨幹的 word embedding 層施加對抗擾動（eps=1.0），提升模型對語言表面變異的魯棒性。

## 模型架構與對應程式碼

我採用了哈工大訊飛聯合實驗室（HFL）發布的四種中文預訓練模型作為骨幹。以下四個 Notebook 獨立執行完整的 5 折交叉驗證訓練流程，並分別產出對應的預測機率矩陣：

| Notebook 檔名 | 預訓練模型 ID | 產出檔案 | 備註 |
| :--- | :--- | :--- | :--- |
| `VeriPromiseESG_2026_large.ipynb` | `hfl/chinese-macbert-large` | `probs_large.pkl` | 隱藏層大小 1024 |
| `VeriPromiseESG_2026_lert.ipynb` | `hfl/chinese-lert-base` | `probs_lert.pkl` | 隱藏層大小 768 |
| `VeriPromiseESG_2026_macbert.ipynb` | `hfl/chinese-macbert-base` | `probs_ensemble.pkl` | 採 3 組 Seed 訓練 15 折平均 |
| `VeriPromiseESG_2026_roberta.ipynb` | `hfl/chinese-roberta-wwm-ext` | `probs_roberta.pkl` | 隱藏層大小 768 |

## 執行環境與依賴套件

本專案程式碼設計於 Google Colab 上執行，建議啟用 GPU（T4 或 T4 x2）。

* **程式語言**: Python 3
* **核心套件**:
    * `transformers==4.44.2` (Hugging Face)
    * `torch` (支援 fp16 混合精度加速)
    * `numpy`, `pandas`, `sklearn`, `pickle`

## 訓練策略細節

* **資料切分**: 將官方訓練集（1,000筆）與驗證集（1,000筆）合併為 2,000 筆，並以 `verification_timeline` 欄位作為分層依據進行 5 折 Stratified K-Fold 切分。
* **最佳化與排程**: 使用 AdamW (weight decay 0.01)，搭配 10% Linear Warmup 排程。
* **模型評估**: 每個 epoch 結束後在本地驗證子集上計算加權 Macro-F1，保留分數最高的 epoch 對應的機率矩陣，最終將各折結果平均作為該模型的輸出，避免過擬合。
