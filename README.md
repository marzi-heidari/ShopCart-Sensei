# 🧠 ShopCart Sensei: Intelligent Product Recommendation Engine

**A context-aware, cold-start-resilient recommender system for e-commerce personalization.**

---

## 🧭 Problem Statement

Traditional e-commerce recommendation systems:
- Struggle with **cold-start** scenarios (new users or products)
- Lack **personalized** suggestions based on recent browsing context
- Often ignore **user sessions**, treating interactions statically

This project explores how to design a **modern recommender system** that addresses these limitations.

---

## 🎯 Project Goal

Develop a **hybrid recommendation engine** that:
- Learns user preferences over time (long-term personalization)
- Understands context in real-time (session-based recommendations)
- Ranks products for:
  - Homepage feed
  - Product detail → similar item suggestions
  - Personalized search result re-ranking

---

### Use Case:
Design a recommendation engine that can suggest relevant products to users based on their browsing sessions, even when user or product history is sparse (cold-start).

### Dataset:
- Source: [RetailRocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Includes:
  - User events (`view`, `addtocart`, `transaction`)
  - Timestamped interactions
  - Product metadata (`item_properties_part_1.csv`, `item_properties_part_2.csv`)
  - Category hierarchy

### Project Goals:
- Understand user behavior from event sequences
- Enrich events with product metadata
- Prepare structured input for training and evaluating recommendation models

---


## 🔧 Features

- Transformer-based sequence models: SASRec and BERT4Rec
- Metrics: `Recall@K`, `NDCG@K`, `MAP@K`
- Cold-start and category-level evaluation
- REST API with FastAPI: `/recommend` and `/similar`
- Streamlit dashboard for engagement and A/B analysis
- SQLite mock database for users and items
- Dockerized for easy deployment
- GitHub Actions CI pipeline

---



## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the dataset
```bash
python preprocessing.py
```

### 3. Train models
```bash
python train_sequence_models.py
```

### 4. Evaluate performance
```bash
python sequence_eval.py \
  --model-path models/sasrec.pt \
  --model-name sasrec \
  --save-metrics results/sasrec.json
```

### 5. Run the API server
```bash
uvicorn api_service:app --reload --port 8000
```

### 6. Run the dashboard
```bash
streamlit run dashboard.py
```

---

## 📦 Docker

### Build and run
```bash
docker build -t rec-api .
docker run -p 8000:8000 rec-api
```

---

## 🔍 Endpoints

### `/recommend?user_id=123&top_k=10`
Return top-k recommendations for a given user based on session history.

### `/similar?product_id=456&top_k=10`
Return top-k similar items to a given product.

---

## ✅ CI/CD

- GitHub Actions runs tests, linter, and Docker build on each push
- Automated checks in `.github/workflows/ci.yml`

---

## 📊 Dashboard Metrics

- Model evaluation metrics over time
- Clicks and conversions
- Bandit vs. static strategy comparisons

---

## 📚 References

- [SASRec](https://arxiv.org/abs/1808.09781)
- [BERT4Rec](https://arxiv.org/abs/1904.06690)
- [RecBole](https://github.com/RUCAIBox/RecBole)

---



## 🧑‍💻 Author

Created by **Marzi Heidari**  
Contact: [LinkedIn](https://www.linkedin.com/in/marzi-heidari) )

---

## 📄 License

MIT License


