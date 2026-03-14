# ITSM Ticket Classification & Predictive Analytics

## Business Objective

This project applies machine learning techniques to historical **IT Service Management (ITSM)** data to improve operational efficiency, risk identification, and workload planning.

Using structured incident and change management data, multiple predictive models were developed to support:

* High-priority incident prediction
* Incident volume forecasting
* Auto-tagging and smart routing
* Change failure prediction

These predictive capabilities help IT operations teams proactively manage incidents and infrastructure changes.

---

## Dataset Description

The dataset contains ITSM ticket-level information including:

* Incident ID
* Impact
* Urgency
* Priority
* Assignment group
* Category
* Reassignment count
* Open and close timestamps
* Change request details

This structured operational data is used to engineer predictive features and train machine learning models.

---

## Exploratory Data Analysis

Key insights discovered during analysis:

* **Priority distribution imbalance** exists in the dataset
* Logical inconsistencies between **Impact and Urgency** influence priority assignment
* Higher **reassignment counts increase resolution time**
* Certain **configuration items have higher change failure rates**

These insights guided the feature engineering and modeling strategy.

---

## Models Built

### High Priority Incident Prediction

**Type:** Multi-class classification
**Goal:** Identify critical incidents early to improve SLA compliance.

---

### Incident Volume Forecasting

**Type:** Time series forecasting
**Goal:** Predict incident workload for better resource planning.

---

### Auto-Tagging & Smart Routing

**Type:** Multi-class classification using structured ticket attributes
**Goal:** Automatically predict assignment group or category to reduce manual ticket triaging.

---

### Change Failure Prediction

**Type:** Binary classification
**Goal:** Identify high-risk changes and reduce production outages.

---

## Tech Stack

### Programming

* Python

### Data Processing & Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Time Series Forecasting

* SARIMA (Statsmodels)
* Prophet

### Data Visualization

* Matplotlib
* Seaborn

### Database

* SQL

---

## Business Impact

* Early identification of high-priority incidents improves **SLA adherence**
* Incident forecasting improves **workforce planning and capacity management**
* Automated routing reduces **manual triaging effort**
* Change risk prediction helps **reduce production outages**
