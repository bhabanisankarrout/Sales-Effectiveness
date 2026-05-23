# 📈 Sales Effectiveness – Lead Category Prediction

A data science project for **FicZon**, aimed at improving sales force effectiveness by replacing manual lead categorization with an ML-powered system that automatically classifies leads as **High Potential** or **Low Potential**.

---

## 📌 Problem Statement

FicZon's business is heavily dependent on sales force effectiveness. As the market matures and new competitors enter, FicZon is experiencing a dip in sales. Lead quality assessment is currently based on manual categorization, making it subjective and dependent on individual sales staff.

The existing quality process provides value only in post-analysis, not during live conversations with prospects.

### Project Goals
1. **Data exploration insights** — Understand sales effectiveness drivers across sources, agents, locations, and time.
2. **ML model** — Predict lead category (High Potential vs. Low Potential) to enable proactive, data-driven sales prioritization.

---

## 📂 Dataset

| File | Description |
|---|---|
| `sales_datas.csv` | Single CSV (semicolon-delimited) containing all lead records |

### Key Features

| Column | Description |
|---|---|
| `Source` | Marketing/lead source channel (Call, Website, Live Chat, etc.) |
| `Sales_Agent` | Assigned sales agent |
| `Location` | Geographic location of the lead |
| `Delivery_Mode` | Product delivery mode |
| `Status` | Current lead status (CONVERTED, POTENTIAL, JUNK LEAD, LOST, etc.) |
| `Mobile` | Lead's mobile number |
| `EMAIL` | Lead's email address |
| `Created` | Timestamp of lead creation |
| `Product_ID` | Product the lead is interested in |

### Target Variable (Engineered)

`Lead_Category` — derived from `Status`:
- **High Potential** → `CONVERTED`, `POTENTIAL`, `IN PROGRESS POSITIVE`
- **Low Potential** → all other statuses (JUNK LEAD, NOT RESPONDING, LOST, etc.)

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

**Univariate Analysis**
- Dominant lead sources: `Call` (~2550), `Live Chat – Direct` (~1850), `Website` (~1600)
- Lead volume heavily concentrated in a few locations (Bangalore, Chennai, Other)
- Top agents drive disproportionately more leads; tail agents handle under 50

**Bivariate Analysis (vs. Status)**
- Email Campaigns and Existing Customers yield the highest conversion proportions
- Sales Agents 12, 6, and 7 show highest conversion rates
- Bangalore, Hyderabad, and Mumbai are strongest converting cities
- Delivery Mode 4 has the highest conversion; Mode 5 has the most junk/lost leads

**Conversion Rate Analysis**
- Highest converting locations: UK (~29%), Bangalore (~22%), UAE (~21%)
- Lowest: Mumbai (~6%), Pune (~7%)

**Time-Based Analysis**
- Monday generates the most leads (~1500); Sunday the fewest (~400)
- Peak lead activity hours: 10 AM–12 PM
- Very low activity before 9 AM and after 6 PM

**Contact Quality Check**
- Mobile numbers: 0% valid (data quality issue — likely stored in non-standard format)
- Email addresses: 65.2% valid

### 2. Data Preprocessing

- Removed duplicate records
- Engineered `valid_mobile` and `valid_email` binary flags from raw contact fields
- Extracted time features from `Created`: `Hour`, `DayOfWeek`, `Month`
- Defined the binary target `Lead_Category` from `Status`
- Applied `LabelEncoder` to all categorical columns
- Applied `StandardScaler` to all features before model training
- Dropped rows with null values in the ML feature set

### 3. Model Building

Trained and compared 6 classification models on an 80/20 stratified train-test split:

| Model | Metric Used |
|---|---|
| **Gradient Boosting** ✅ | Best ROC-AUC & CV Accuracy |
| Random Forest | — |
| Logistic Regression | — |
| Decision Tree | — |
| SVM | — |
| Naive Bayes | — |

> Gradient Boosting was selected as the best model based on ROC-AUC and 5-fold cross-validation accuracy.

### 4. Feature Importance (Gradient Boosting)

Top predictive features:
- `Source` — lead channel is the strongest signal
- `Sales_Agent` — agent assignment has significant impact
- `Location` — geography drives conversion likelihood
- `Hour` — time of lead creation carries behavioral signal
- `valid_email` — contact completeness is a quality proxy
- `Delivery_Mode`, `DayOfWeek`, `valid_mobile`

### 5. Hyperparameter Tuning

Two-stage tuning process:

**Stage 1 — RandomizedSearchCV** (50 iterations, 5-fold Stratified CV, `roc_auc` scoring)
- Searched across `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `max_features`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`

**Stage 2 — GridSearchCV** (fine-tuned around the best RandomizedSearch parameters)
- Narrowed the search space for precision tuning

### 6. Threshold Optimization

Tested classification thresholds from 0.10 to 0.85 and plotted Precision / Recall / F1 tradeoff curves to identify the optimal threshold for maximizing F1 on the **High Potential** class — ensuring the model surfaces valuable leads without flooding the sales team with false positives.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes)
- **Tuning:** `RandomizedSearchCV`, `GridSearchCV`, `StratifiedKFold`

---

## 📁 Project Structure

```
sales-effectiveness/
│
├── sales_datas.csv                    # Raw dataset
├── Sales_Effectiveness.ipynb          # Main analysis and modeling notebook
├── bivariate_stacked_bar.png          # Status proportion charts per feature
├── bivariate_heatmaps.png             # Crosstab heatmaps
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

1. Clone this repository
2. Place `sales_datas.csv` in the project root
3. Open and run `Sales_Effectiveness.ipynb` in Jupyter

---

## 💡 Key Insights Summary

| Area | Insight |
|---|---|
| **Lead Source** | Call and Live Chat – Direct generate the most volume; Email Campaigns convert the best |
| **Location** | UK and Bangalore have the highest conversion rates; Mumbai and Pune lag significantly |
| **Sales Agents** | Agents 12, 6, and 7 are top converters; workload is unevenly distributed |
| **Timing** | Monday mornings (10 AM–12 PM) are peak hours for lead generation |
| **Contact Quality** | ~35% of emails are invalid; mobile number data needs cleaning |
| **Delivery Mode** | Mode 4 is the most effective; Mode 5 has the highest junk/lost lead ratio |

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| Mobile numbers stored in non-standard format (0% valid) | Engineered a `valid_mobile` binary flag as a data quality proxy |
| Manual and subjective lead categorization | Derived a binary `Lead_Category` target from existing `Status` field |
| Class imbalance between High and Low Potential leads | Used stratified train-test split and ROC-AUC as the primary evaluation metric |
| Finding the right classification threshold | Performed threshold sweep (0.10–0.85) to optimize F1 for High Potential class |

---

## 📈 Future Improvements

- Integrate NLP on lead notes/chat transcripts for richer signal
- Build a real-time lead scoring API for CRM integration
- Add SHAP explainability so sales agents understand why a lead is flagged High Potential
- Periodic model retraining as new lead data accumulates
- Fix mobile number data pipeline to restore contact quality signal
