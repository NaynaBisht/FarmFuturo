# FARM FUTURO - Machine Learning-Based Crop Recommendation System

**Farm Futuro** is a machine learning project designed to help farmers make informed decisions about crop selection based on their location, crop preferences, and local environmental factors such as temperature, humidity, rainfall, and soil pH. This system leverages decision tree algorithms to analyze agricultural data and provides precision recommendations, enhancing sustainable farming practices and optimizing crop productivity.

---

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement and Objective](#problem-statement-and-objective)
- [Dataset Details](#dataset-details)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Algorithm Analysis and Results](#algorithm-analysis-and-results)
- [Visualizations](#visualizations)
- [Conclusion and Future Scope](#conclusion-and-future-scope)
- [Installation and Implementation](#installation-and-implementation)

---

## Introduction

Agriculture faces increasing challenges, including unpredictable climate changes and the need for sustainable practices. Farm Futuro addresses these by providing an ML-based solution to predict the most suitable crops for a farmer's specific conditions. It uses advanced algorithms to analyze regional soil and climate data, allowing farmers to make data-informed crop selections that can enhance yields and contribute to sustainable agriculture.

---

## Problem Statement and Objective

Farmers often lack the tools needed to choose crops that are both viable for their land and environmentally suitable. This project aims to tackle issues such as soil degradation, unsustainable crop patterns, and inefficient resource use by developing a system that:
- Analyzes geographic and environmental factors affecting crop selection.
- Provides crop recommendations based on historical data and real-time inputs.
- Encourages sustainable practices by aligning crop selection with soil and climate requirements.

---

## Dataset Details

- **Source:** The dataset used for this project comes from the National Institute of Agricultural Extension Management (MANAGE) and the Ministry of Agriculture and Farmers Welfare, India.
- **Volume:** Approximately 2,200 observations across 23 regions and 22 major crops.
- **Features:**
  - State (geographic location)
  - Crop Type (cereals, pulses, fruits, etc.)
  - Environmental factors (rainfall, temperature, humidity, soil pH)
  - Crop suitability and seasonal preferences

---

## Data Preprocessing

- **Data Extraction:** The dataset has been curated from surveys and observational studies on agriculture.
- **Data Cleaning:** Missing values were handled and state names updated. Using the Pandas library, data inconsistencies were corrected.
- **Data Sorting:** Data is sorted by state and crop type, allowing for region-specific recommendations.

---

## Machine Learning Model

Farm Futuro uses a **Decision Tree Classifier** for crop recommendation:
1. **Feature Inputs:** Temperature, humidity, soil pH, rainfall, crop type, and state.
2. **Encoding:** Categorical features are label-encoded for model compatibility.
3. **Training:** Data is split into training and testing sets. The decision tree algorithm learns from the historical data to predict suitable crops.
4. **Prediction:** Based on user inputs, the trained model recommends the most suitable crops, achieving an accuracy of 90-95%.

---

## Algorithm Analysis and Results

The decision tree classifier provides robust crop recommendations that align well with traditional crop patterns in each region. The algorithm achieved high accuracy, supporting its efficacy in providing actionable insights.

---

## Visualizations

Data visualizations play a key role in Farm Futuro, aiding in understanding crop distributions across states. Key plots include:
- **Crop distribution by state**
- **Division-wise crop distribution**
- **Histograms for crop counts and soil pH**
- **Pie charts for regional crop diversity**

---

## Conclusion and Future Scope

Farm Futuro effectively enhances crop decision-making by offering tailored recommendations based on environmental and geographic factors. Future improvements may include:
- **Crop Rotation Prediction:** Advising farmers on crop rotation to maintain soil health and pH balance.
- **Expanded Dataset Integration:** Adding data from new regions to enhance the model’s generalizability.

Farm Futuro's successful implementation promises to drive better agricultural outcomes and supports farmers in navigating climate challenges.

---

## Installation and Implementation

### Prerequisites
- Python 3.x
- Required libraries: `pandas`, `scikit-learn`, `matplotlib`, `numpy`

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/FarmFuturo.git
2. **Navigate to the project directory:**
   ```bash
   cd FarmFuturo
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the project:**
   ```bash
   apiproject.py

---

## Get Involved

Farm Futuro is open to contributions! If you're passionate about leveraging technology for sustainable agriculture or have ideas for new features, feel free to open an issue or submit a pull request. Together, we can refine and expand Farm Futuro to serve an even broader range of agricultural needs and regions.

Farm Futuro isn’t just a tool—it’s a step towards a more sustainable and data-driven future in agriculture. By combining advanced machine learning with essential environmental insights, we aim to empower farmers and promote smarter, eco-friendly farming practices. We hope that Farm Futuro serves as a valuable resource and sparks innovation for a brighter agricultural future.

