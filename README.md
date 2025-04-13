# ShopCart Sensei: Intelligent Product Recommendation Engine


**A context-aware, cold-start-resilient recommender system for e-commerce personalization — built with Shopify-scale in mind.**

---

##  Problem Statement

Traditional e-commerce recommendation systems:
- Struggle with **cold-start** scenarios (new users or products)
- Lack **personalized** suggestions based on recent browsing context
- Often ignore **user sessions**, treating interactions statically

This project explores how to design a **modern recommender system** that addresses these limitations — optimized for real-world Shopify-style storefronts.

---

## Project Goal

Develop a **hybrid recommendation engine** that:
- Learns user preferences over time (long-term personalization)
- Understands context in real-time (session-based recommendations)
- Ranks products for:
  - Homepage feed
  - Product detail → similar item suggestions
  - Personalized search result re-ranking

---


## Dataset

You can use one of the following:
- [RetailRocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [Amazon Reviews](https://nijianmo.github.io/amazon/index.html)
- Or synthetic data (for full control during development)

Features include:
- User ID, session ID, timestamps
- Product metadata: title, category, price, image, description
- Click/view/purchase events

---

## Hypothesis

> A recommendation engine that adapts to user context and learns personalized preferences over time will significantly outperform static or global models in terms of engagement and relevance.

---



## 🛠️ Tech Stack

| Component | Tool |
|----------|------|
| Modeling | PyTorch, HuggingFace Transformers, LightFM |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Bandits | Vowpal Wabbit / LinUCB |
| Data Handling | Pandas, NumPy |
| Deployment | Docker |

---

## 👤 Author

Marzi Heidari  
PhD Candidate, Machine Learning  [LinkedIn](https://www.linkedin.com/in/marzi-heidari/) | [GitHub](#)

---

## 📜 License

MIT

