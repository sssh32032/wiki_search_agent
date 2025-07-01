# Wikipedia Assistant - 智慧百科助手

基於 RAG (Retrieval-Augmented Generation) 技術的智慧百科助手，整合 Wikipedia API、向量資料庫和大型語言模型，提供準確、即時的知識問答服務。

## 🚀 快速進入專案環境

**⚠️ 重要：必須使用 cmd（命令提示字元），不要使用 PowerShell！**

```cmd
# 1. 開啟 cmd 並切換到專案目錄
cd C:\Users\sssh3\Desktop\side_project\agent_exercise\wiki_search

# 2. 檢查虛擬環境
ls "C:\Users\sssh3\AppData\Local\pypoetry\Cache\virtualenvs"

# 3. 啟動虛擬環境（選擇最新的）
C:\Users\sssh3\AppData\Local\pypoetry\Cache\virtualenvs\rag-exercise-kCDYDLwJ-py3.12\Scripts\activate.bat

# 4. 驗證環境
python -c "from app.config import settings; print('✅ 配置載入成功')"
```

**成功標誌**：命令提示字元前面出現 `(rag-exercise-py3.12)` 前綴

## 🎯 專案目標

### 核心功能
- **智慧問答**：基於 Wikipedia 資料的準確回答
- **即時檢索**：快速從向量資料庫中檢索相關資訊
- **多語言支援**：支援中文和英文查詢
- **安全驗證**：使用 Guardrails 確保輸出品質
- **API 服務**：提供 RESTful API 介面

### 技術特色
- **RAG 架構**：結合檢索和生成的混合架構
- **向量檢索**：使用 FAISS 進行高效相似度搜尋
- **重排序優化**：使用 Cohere Rerank 提升檢索準確度
- **模組化設計**：清晰的模組分離，易於維護和擴展

## 🏗️ 技術架構

### 核心技術棧
- **後端框架**：FastAPI
- **語言模型**：OpenAI GPT-4o-mini
- **向量化模型**：OpenAI text-embedding-3-small
- **重排序模型**：Cohere Rerank
- **向量資料庫**：FAISS
- **資料來源**：Wikipedia API
- **輸出驗證**：Guardrails AI

### 系統架構
```
Wikipedia API → 資料獲取 → 文本切片 → 向量化 → FAISS 儲存
                                                      ↓
用戶查詢 → 向量檢索 → 重排序 → LLM 生成 → Guardrails 驗證 → 回應
```

## 📁 專案結構

```
wiki_search/
├── pyproject.toml           # Poetry 相依管理
├── README.md                # 專案說明文件
├── .env                     # 環境變數設定
├── env.example              # 環境變數範例
├── data/                    # 臨時儲存 Wikipedia 資料
│   └── wikipedia_pages_*.json  # Wikipedia 資料檔案
├── scripts/                 # 資料處理腳本
│   ├── fetch_wiki.py        # Wikipedia 爬取、清理、儲存
│   └── build_embeddings.py  # 切片、向量化並存入 Vector Database
├── app/                     # 核心應用程式碼
│   ├── api/                 # FastAPI 路由與服務
│   │   ├── main.py          # 啟動應用
│   │   └── routes.py        # 定義 /query 路由
│   ├── core/                # 核心邏輯
│   │   ├── retriever.py     # 檢索模組，整合 Vector Store + Reranking
│   │   ├── llm_chain.py     # 串接 LLM 生成流程
│   │   ├── guardrail.py     # 輸出驗證邏輯
│   │   └── wiki_tool.py     # 自定義 Wikipedia Tool
│   └── config.py            # 環境設定、API 金鑰
├── faiss_index/             # 儲存向量資料庫檔案
│   ├── faiss_index_*.index  # FAISS 向量索引檔案
│   └── metadata_*.json      # 文本切片元資料
├── logs/                    # 日誌檔案
│   ├── embeddings.log       # 向量化處理日誌
│   └── wiki_fetch.log       # Wikipedia 資料獲取日誌
└── Dockerfile               # 容器化部署設定
```

## 📊 開發進度

### ✅ 已完成
- [x] **環境設定與基礎架構**
  - Poetry 環境設定
  - 依賴套件安裝（FastAPI, OpenAI, Cohere, FAISS, LangChain 等）
  - 專案目錄結構建立
  - 環境變數管理系統 (`app/config.py`)
  - `.env` 檔案設定
- [x] **Wikipedia 資料獲取模組** (`scripts/fetch_wiki.py`)
  - 繁體中文 Wikipedia 資料爬取
  - 自動簡體轉繁體功能
  - 資料清理與 JSON 格式儲存
  - 日誌記錄系統
- [x] **向量化處理模組** (`scripts/build_embeddings.py`)
  - 智能文本切片（以句子為單位）
  - 使用 sentence-transformers 進行向量化
  - FAISS 向量索引建立與儲存
  - 元資料管理（包含原始文本內容）
  - 相似度搜尋功能

### 🔄 進行中
- [ ] **核心模組開發**
  - [x] 檢索與重排序模組 (`app/core/retriever.py`) - ✅ 已完成
  - [ ] LLM 生成模組 (`app/core/llm_chain.py`)
  - [ ] Guardrails 安全驗證模組 (`app/core/guardrail.py`)

### 📋 待開發
- [ ] **API 整合**
  - [ ] FastAPI 路由設計 (`app/api/routes.py`)
  - [ ] API 端點實作 (`app/api/main.py`)
- [ ] **測試與優化**
  - [ ] 功能測試
  - [ ] 效能優化
  - [ ] 錯誤處理
- [ ] **部署與文檔**
  - [ ] Docker 容器化
  - [ ] API 文檔
  - [ ] 使用說明

## 🚀 快速開始

### 環境需求
- Python 3.12+
- Poetry

### 安裝步驟
1. **克隆專案**
   ```bash
   git clone <repository-url>
   cd RAG_exercise
   ```

2. **安裝依賴**
   ```bash
   poetry install
   ```

3. **設定環境變數**
   ```bash
   cp env.example .env
   # 編輯 .env 檔案，填入你的 API 金鑰
   ```

4. **進入 Poetry 虛擬環境（Windows 詳細步驟）**

   **⚠️ 重要：必須使用 cmd（命令提示字元），不要使用 PowerShell！**

   1. **開啟 cmd（命令提示字元）**
      - 方法一：在「開始」選單搜尋「cmd」並開啟
      - 方法二：在 PowerShell 中輸入 `cmd` 切換到命令提示字元
      - 方法三：按 `Win + R`，輸入 `cmd` 後按 Enter

   2. **切換到專案目錄**
      ```cmd
      cd C:\Users\sssh3\Desktop\side_project\agent_exercise\wiki_search
      ```

   3. **檢查可用的虛擬環境**
      ```cmd
      ls "C:\Users\sssh3\AppData\Local\pypoetry\Cache\virtualenvs"
      ```
      - 會顯示類似以下的結果：
      ```
      rag-exercise-kCDYDLwJ-py3.12
      rag-exercise-wb-tVZr8-py3.12
      ```
      - **選擇最新的虛擬環境**（通常是時間戳較新的，檔案夾修改時間較新的）

   4. **啟動 Poetry 虛擬環境**
      ```cmd
      C:\Users\sssh3\AppData\Local\pypoetry\Cache\virtualenvs\rag-exercise-kCDYDLwJ-py3.12\Scripts\activate.bat
      ```
      - **成功標誌**：命令提示字元前面會出現 `(rag-exercise-py3.12)` 或類似的前綴
      - 例如：`(rag-exercise-py3.12) C:\Users\sssh3\Desktop\side_project\agent_exercise\wiki_search>`

   5. **驗證環境是否正確**
      ```cmd
      python --version
      ```
      - 應該顯示：`Python 3.12.10`

   6. **測試配置載入**
      ```cmd
      python -c "from app.config import settings; print('✅ 配置載入成功')"
      ```
      - 如果顯示 `✅ 配置載入成功`，表示環境設定完全正確

   **🔧 故障排除：**
   - 如果看到 `ModuleNotFoundError`，表示還沒進入正確的虛擬環境
   - 如果 `activate.bat` 執行後沒有出現括號前綴，請重新執行步驟 4
   - 如果不確定虛擬環境路徑，可用 `poetry env info --path` 查詢（但需要在 PowerShell 中執行）

5. **測試基本功能**
   ```bash
   # 測試配置
   python -c "from app.config import settings; print('✅ 配置載入成功')"
   
   # 測試 Wikipedia 資料獲取
   python scripts/fetch_wiki.py
   
   # 測試向量化處理
   python scripts/build_embeddings.py
   ```

### API 金鑰設定
在 `.env` 檔案中設定以下 API 金鑰：
- `OPENAI_API_KEY`：OpenAI API 金鑰
- `COHERE_API_KEY`：Cohere API 金鑰

## 🔧 開發指南

### 開發順序
1. **✅ 資料獲取**：`scripts/fetch_wiki.py` - 已完成
2. **✅ 向量化處理**：`scripts/build_embeddings.py` - 已完成
3. **✅ 檢索系統**：`app/core/retriever.py` - 已完成
4. **📋 生成系統**：`app/core/llm_chain.py` - 待開發
5. **📋 安全驗證**：`app/core/guardrail.py` - 待開發
6. **📋 API 整合**：`app/api/` - 待開發

### 測試
```bash
# 測試 Wikipedia 資料獲取
python scripts/fetch_wiki.py

# 測試向量化處理
python scripts/build_embeddings.py

# 測試配置
python -c "from app.config import settings; print('✅ 配置載入成功')"

# 測試檢索器模組
python app/core/retriever.py

# 測試 API（待開發）
# uvicorn app.api.main:app --reload
```

## 📈 未來規劃

### 短期目標 (1-2 週)
- [x] 完成 Wikipedia 資料獲取功能
- [x] 完成向量化處理與儲存功能
- [ ] 完成核心 RAG 功能（檢索 + 生成）
- [ ] 建立基本 API 端點
- [ ] 實現 Wikipedia 資料自動更新

### 中期目標 (1 個月)
- [ ] 優化檢索準確度
- [ ] 增加多語言支援
- [ ] 實作快取機制
- [ ] 完善錯誤處理

### 長期目標 (2-3 個月)
- [ ] 支援更多資料來源
- [ ] 實作用戶管理系統
- [ ] 增加對話歷史功能
- [ ] 部署到雲端服務

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

### 開發規範
- 使用 Poetry 管理依賴
- 遵循 PEP 8 程式碼風格
- 撰寫適當的註解和文檔
- 新增測試案例

## 📄 授權

MIT License

## 📞 聯絡資訊

如有問題或建議，請提交 Issue 或聯絡開發團隊。

# 虛擬環境啟動與切換說明

## 1. 進入 Poetry 全域虛擬環境

Poetry 安裝時會自動建立一個全域虛擬環境，通常只建議用來執行 poetry 指令本身，不建議安裝專案套件。

- **全域虛擬環境路徑**：
  - `C:\Users\sssh3\AppData\Roaming\pypoetry\venv`
- **啟動方式（不建議用於專案開發）**：
  1. 開啟 cmd
  2. 執行：
     ```