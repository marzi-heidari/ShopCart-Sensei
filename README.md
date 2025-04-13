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

## 🧰 Step 1: Define the Use Case and Dataset

This step sets the foundation of the project by clearly outlining the recommendation scenario and selecting an appropriate dataset.

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

## 🧹 Step 2: Data Preprocessing & Feature Engineering

This step transforms raw interaction logs and item metadata into clean, sessionized data suitable for model training.

### Key Features:
- **Chunked item processing**: Handles large `item_properties_part_1.csv` and `item_properties_part_2.csv` efficiently using streamed chunks to avoid memory overload.
- **Session generation**: Users are assigned session IDs based on a 30-minute inactivity threshold.
- **Feature encoding**: Converts `visitorid`, `itemid`, `categoryid`, and `brand` into numerical form using `LabelEncoder`.
- **Safe merging**: Event data is merged with the latest item properties (one per `(itemid, property)`), then sorted chronologically.

### Output:
- A memory-efficient Parquet file: `data/processed/processed_sessions.parquet`
- Format: one row per user interaction including session ID, timestamp, item, event type, category, and brand.

### Run the script:
```bash
python preprocessing.py
```

