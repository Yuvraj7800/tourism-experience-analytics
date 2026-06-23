# 🌍 Tourism Experience Analytics
### Classification · Prediction · Recommendation System

A complete end-to-end Machine Learning project that analyzes **52,930 tourism transactions** across **33,530 users** to predict attraction ratings, classify visitor behavior, and generate personalized attraction recommendations — deployed as an interactive Streamlit web app.

🔗 **Live App:** _Add your Streamlit Cloud link here after deployment_

---

## 📋 Problem Statement

Tourism agencies and travel platforms need data-driven tools to enhance user experience by providing personalized recommendations, predicting user satisfaction, and classifying visitor behavior. This project solves that with three ML pipelines: **Regression**, **Classification**, and **Recommendation**.

---

## 🎯 Objectives

| # | Task | Goal |
|---|------|------|
| 1 | **Regression** | Predict the rating (1–5) a user will give to an attraction |
| 2 | **Classification** | Predict the visit mode (Business, Couples, Family, Friends, Solo) |
| 3 | **Recommendation** | Suggest personalized attractions via Collaborative + Content-Based filtering |

---

## 📊 Dataset

| Metric | Value |
|---|---|
| Total Transactions | 52,930 |
| Unique Users | 33,530 |
| Attractions | 30 |
| Attraction Types | 17 |
| Overall Avg Rating | 4.158 / 5 |
| Source Tables | 9 (Transaction, User, Item, Mode, Type, City, Country, Region, Continent) |

---

## 🏆 Model Results

### Regression — Predicting Attraction Ratings
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.5034 | 0.2950 | 0.7341 |
| Random Forest | 0.5449 | 0.2788 | 0.6885 |
| XGBoost | 0.4977 | 0.2674 | 0.7401 |
| **LightGBM (Best)** | **0.4974** | **0.2647** | **0.7405** |

### Classification — Predicting Visit Mode
| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest | 48.6% | 0.4754 |
| XGBoost | 49.3% | 0.4401 |
| **LightGBM (Best Accuracy)** | **51.3%** | 0.4695 |
| Gradient Boosting | 50.1% | 0.4533 |

> Note: Random Forest has the highest F1-score (0.4754) while LightGBM has the highest raw accuracy (51.3%) — both are reported in the notebook for comparison.

**Per-class performance (Random Forest):**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Couples | 0.53 | 0.63 | 0.57 |
| Family | 0.50 | 0.50 | 0.50 |
| Friends | 0.38 | 0.30 | 0.33 |
| Solo | 0.38 | 0.23 | 0.28 |
| Business | 0.44 | 0.22 | 0.30 |

### Recommendation System — SVD Collaborative Filtering
| Metric | Score |
|---|---|
| RMSE (held-out pairs) | 0.2290 |
| MAE (held-out pairs) | 0.0646 |

> Full model comparison, residual analysis, feature importance, and confusion matrices are available in `Tourism_Analytics.ipynb`.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy, OpenPyXL
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Notebook:** Jupyter

---

## 🖥️ Streamlit App — 6 Interactive Pages

| Page | Description |
|---|---|
| 🏠 Home Dashboard | KPIs, top attractions, top regions, user segments |
| 📊 EDA Explorer | Geography, Ratings, Visit Modes, Time Trends (4 tabs) |
| ⭐ Predict Rating | Input details → live rating prediction |
| 🧳 Predict Visit Mode | Input details → live visit mode prediction with confidence chart |
| 🎯 Get Recommendations | Collaborative, Content-Based & Hybrid recommendations |
| 📈 Model Performance | Feature importance, classification reports |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tourism-experience-analytics.git
cd tourism-experience-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

App opens at `http://localhost:8501`

To explore the full analysis notebook:
```bash
pip install jupyter
jupyter notebook Tourism_Analytics.ipynb
```

---

## 📁 Project Structure

```
tourism-experience-analytics/
├── app.py                              # Streamlit application (6 pages)
├── Tourism_Analytics.ipynb             # Full analysis: cleaning → EDA → models → insights
├── Tourism_Cleaned_Dataset.csv         # Final preprocessed dataset (52,930 × 27)
├── Tourism_Report.docx                 # Documentation report
├── Tourism_Analytics_Presentation.pptx # Project presentation slides
├── requirements.txt                    # Python dependencies
├── Transaction.xlsx                    # Core visit/rating data
├── User.xlsx                           # User demographics
├── Item.xlsx                           # Attraction details
├── Mode.xlsx                           # Visit mode lookup
├── Type.xlsx                           # Attraction type lookup
├── City.xlsx / Country.xlsx / Region.xlsx / Continent.xlsx   # Geographic hierarchy
└── README.md
```

---

## 💡 Key Business Insights

1. **Couples** are the largest visitor segment — opportunity for targeted couple packages
2. **AttractionAvgRating** and **UserAvgRating** are the strongest predictors of satisfaction
3. A small number of attractions account for a large share of total visits — diversify marketing
4. Visit volume shows clear seasonal peaks — useful for promotional timing
5. The recommendation system's RMSE of 0.23 indicates strong real-world usability

---

## 📈 Future Improvements

- Address class imbalance in visit mode classification (SMOTE / class weighting)
- Incorporate the `Updated_Item.xlsx` attraction dataset for richer content-based features
- Add deep learning–based sequence models for visit pattern prediction
- Deploy a live Power BI dashboard alongside the Streamlit app

---

## 👤 Author

Project completed as part of a Data Science / Machine Learning portfolio — Tourism domain.

**Skills demonstrated:** Data Cleaning, EDA, Feature Engineering, Regression, Classification, Recommendation Systems, Model Evaluation, Streamlit Deployment.
