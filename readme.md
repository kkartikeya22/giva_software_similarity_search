# 📄 Document Similarity Search API

🚀 A FastAPI-based API for finding similar documents using **FAISS** and **sentence-transformers**.

## 📌 Features
- 🔍 **Search for Similar Documents** using embeddings.
- ⚡ **FAISS Indexing** for fast similarity search.
- 📚 **Supports Cosine & L2 Distance** metrics.
- 🌍 **CORS Enabled** for frontend integration.
- 🏗️ **Deployable on Render**.

---

## 🛠️ **Installation & Setup**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/kkartikeya22/giva_software_similarity_search/blob/main
```

### **2️⃣ Create & Activate Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Run API Locally**
```sh
python -m hypercorn main:app --bind 127.0.0.1:8000
```
📌 **Open in browser:** `http://127.0.0.1:8000/docs`

---

## 🚀 **API Endpoints**

### 🌍 **Live API Link**
🔗 API Base URL :- https://giva-software-similarity-search.onrender.com/

### **Available Routes**
- `GET /health` → Check if the API is running.
- `GET /api/search?q=QUERY&metric=cosine` → Find similar documents.
### **1️⃣ Health Check**
- **URL:** `GET /health`
- **Response:**
  ```json
  {"status": "healthy"}
  ```

### **2️⃣ Search for Similar Documents**
- **URL:** `GET /api/search?q=AI&metric=cosine`
- **Query Parameters:**
  - `q` (required) → Search query text.
  - `metric` (optional) → `cosine` (default) or `l2`.
- **Example Response:**
  ```json
  {
    "query": "AI",
    "results": [
      {"document": "Artificial intelligence is transforming industries.", "score": 0.98},
      {"document": "Machine learning is a subset of AI.", "score": 0.95}
    ]
  }
  ```

---

## 🌍 **Deploy on Render**
### **1️⃣ Update `requirements.txt`**
Make sure it includes:
```
fastapi
hypercorn
sentence-transformers
faiss-cpu
numpy
```
For **GPU support**, use:
```
faiss-gpu
```

### **2️⃣ Start Command on Render**
```sh
hypercorn main:app --bind 0.0.0.0:$PORT
```

### **3️⃣ Set Environment Variables on Render**
- `PORT = 10000` (Render auto-assigns, but this ensures proper binding).

---

## 🛠️ **Troubleshooting**
| **Issue** | **Fix** |
|-----------|--------|
| **500 Internal Server Error** | Check **Render Logs** for the exact error |
| **FAISS Index Failure** | Ensure `documents.txt` is loaded correctly |
| **Missing Dependencies** | Run `pip install -r requirements.txt` & redeploy |
| **Works Locally But Not on Render** | **Restart service & check logs** |

---
🔥 **Developed by Kartikeya Katiyar** 🚀

