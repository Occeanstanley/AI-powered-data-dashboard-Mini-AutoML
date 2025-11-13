# 🧠 AI-Powered Data Dashboard (Mini AutoML)
*A Streamlit-based smart dashboard for data exploration, machine learning, and automatic reporting.*

🔗 **Live Demo:**  
https://huggingface.co/spaces/occeanstanley9/Streamlit  

---

## 🚀 Overview
The **AI-Powered Data Dashboard (Mini AutoML)** is an interactive machine learning tool that allows users to:

- Upload CSV datasets  
- Explore and summarize data  
- Inspect schema and missing values  
- Auto-detect classification vs. regression tasks  
- Train ML models with one click  
- View evaluation metrics  
- Export analysis reports as **Markdown, HTML, or PDF**

Ideal for **students, analysts, and ML beginners** who need a quick, no-code environment for modeling and exploration.

---

## ⭐ Features

### 📁 **1. Data Upload & Preview**
- Upload `.csv` files  
- Preview first 20 rows  
- Auto-detected schema  
- Missingness summary  
- Column type inference  

### 📊 **2. Exploratory Data Analysis**
- Numeric distributions (up to 10 features)  
- Missing value heatmap  
- Data-quality overview  
- Summary statistics  

### 🤖 **3. AutoML Training Workflow**
- Smart target column selector  
- Auto task detection (classification/regression)  
- Train/test split slider  
- Fast mode (3-fold CV)  
- Optional RandomForest GridSearchCV  
- Evaluation metrics leaderboard  

### 📝 **4. Report Exporting**
Export complete analysis results as:
- `report.md` (Markdown)  
- `report.html` (HTML)  
- `report.pdf` (PDF with visuals and metrics)  

---

## 🛠️ Tech Stack

| Component      | Technology          |
|----------------|---------------------|
| Interface      | Streamlit           |
| Backend        | Python              |
| Machine Learning | scikit-learn     |
| Data Handling  | pandas, numpy       |
| Visualization  | seaborn, matplotlib |
| Reporting      | markdown, fpdf      |

---

## 📂 Project Structure
---

## ▶️ How to Run Locally

### **1. Clone the project**
```bash
git clone https://github.com/Occeanstanley/AI-Powered-Data-Dashboard-Mini-AutoML
cd AI-Powered-Data-Dashboard-Mini-AutoML

pip install -r requirements.txt
streamlit run app.py

