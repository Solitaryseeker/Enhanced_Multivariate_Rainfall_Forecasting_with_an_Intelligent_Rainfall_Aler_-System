# Enhanced Multivariate Rainfall Forecasting Using Deep Learning with an Intelligent Rainfall Alert System

An intelligent rainfall forecasting system that applies deep learning architectures to multivariate time-series data for rainfall prediction and early-warning support. The project implements and compares four deep learning models: **LSTM, GRU, Transformer Encoder, and iTransformer**.

The project also includes a web-based rainfall prediction application focused on Yemen, providing rainfall forecasts, visualizations, and an alert-level interface.

## Project Overview

Accurate rainfall forecasting is important for climate monitoring, agricultural planning, and disaster preparedness. This project explores multivariate time-series forecasting using different deep learning architectures to identify an effective model for rainfall prediction.

The system uses historical rainfall indicators and temporal features to learn rainfall patterns and generate predictions. The model comparison helps evaluate the performance of recurrent and Transformer-based architectures.

## Key Features

- Multivariate time-series rainfall forecasting.
- Implementation of four deep learning architectures.
- Model performance evaluation and comparison.
- Time-series feature engineering and preprocessing.
- Hyperparameter optimization using Optuna.
- Yemen-focused rainfall prediction web application.
- Interactive rainfall visualizations.
- Intelligent rainfall alert-level interface.
- Location-based rainfall prediction support.

## Deep Learning Models

The project implements four models:

| Model | Architecture | Description |
|---|---|---|
| LSTM | Long Short-Term Memory | Learns temporal dependencies in sequential rainfall data. |
| GRU | Gated Recurrent Unit | Uses a gated recurrent architecture for sequence modelling. |
| Transformer Encoder | Self-Attention | Captures relationships between time steps through attention mechanisms. |
| iTransformer | Inverted Transformer | Applies attention across variates to model multivariate time-series relationships. |

## Model Performance Comparison

The following table presents the reported evaluation results of the four models.

| Evaluation Metric | LSTM | GRU | Transformer Encoder | iTransformer |
|---|---:|---:|---:|---:|
| RMSE | 0.0608 | 0.4812 | 0.0897 | **0.0885** |
| MAE | 0.0262 | 0.3291 | 0.0657 | **0.0629** |
| R² Score | 0.4623 | 0.7510 | 0.7409 | **0.7703** |

*Note: Values are reported from the project's model evaluation results. Dashes indicate metrics not supplied in the reported results. Confirm that all models were evaluated on the same test split and target scale before making a strict numerical comparison.*

### Best-Performing Model: iTransformer

The iTransformer achieved the highest reported coefficient of determination (R² = 0.7703) among the four evaluated models. It also reported the lowest RMSE (0.0885) and MAE (0.0629) among the GRU, Transformer Encoder, and iTransformer results.

Based on these reported results, iTransformer was selected as the preferred model for the rainfall prediction application.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / PyTorch |
| Model Optimization | Optuna |
| Visualization | Matplotlib, Plotly |
| Web Application | HTML, CSS, JavaScript, Python |
| Development Environment | Jupyter Notebook, PyCharm, VS Code |
| Version Control | Git, GitHub |

## Project Structure

```text
Enhanced_Multivariate_Rainfall_Forecasting/
│
├── GRU.ipynb
├── LSTM.ipynb
├── Transformer_Encoder.ipynb
├── iTransformer.ipynb
│
├── yemen-rainfall/
│   └── yemen-rainfall/
│       ├── README.md
│       ├── index.html
│       ├── server.py
│       ├── run.bat
│       ├── run.sh
│       ├── requirements.txt
│       ├── best_itransformer .pth
│       ├── model_config.json
│       ├── scalers.json
│       ├── x_scaler.pkl
│       ├── y_scaler.pkl
│       ├── pcode_map.json
│       └── prepare_artifacts.py
│
└── README.md
```

## Yemen Rainfall Prediction Web Application

The project includes a web application designed to demonstrate the application of the selected iTransformer model for rainfall prediction in Yemen.

**Application source code:** [Yemen Rainfall Web Application](https://github.com/Solitaryseeker/Enhanced_Multivariate_Rainfall_Forecasting_with_an_Intelligent_Rainfall_Aler_-System/tree/main/yemen-rainfall)

### Main Application Features

- Rainfall prediction using the selected iTransformer model.
- Location-based prediction interface.
- Rainfall input indicators.
- Interactive rainfall visualization.
- Prediction result display.
- Rainfall alert-level interface.

The alert interface uses project-defined rainfall thresholds. These thresholds should be independently validated against appropriate local standards before operational use.

## Input Features

The web application uses the following input indicators:

- `rfh` – Rainfall indicator.
- `r1h` – Rainfall indicator.
- `r3h` – Rainfall indicator.
- `month` – Month of the year.
- `dekad` – Ten-day period of the month.

The forecasting model uses a 24-step input sequence and predicts the target variable `rfh`.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Solitaryseeker/Enhanced_Multivariate_Rainfall_Forecasting_with_an_Intelligent_Rainfall_Aler_-System.git
```

### 2. Navigate to the Web Application Directory

```bash
cd Enhanced_Multivariate_Rainfall_Forecasting_with_an_Intelligent_Rainfall_Aler_-System/yemen-rainfall/yemen-rainfall
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

On Windows:

```bash
run.bat
```

Alternatively, run:

```bash
python server.py
```

On Linux or macOS:

```bash
chmod +x run.sh
./run.sh
```

Follow the local address displayed by the server in your terminal to open the application.

## Research and Project Objectives

The main objectives of this project are:

1. To investigate deep learning approaches for multivariate rainfall forecasting.
2. To implement and evaluate LSTM, GRU, Transformer Encoder, and iTransformer architectures.
3. To compare model performance using standard regression evaluation metrics.
4. To explore the effectiveness of attention-based architectures for multivariate time-series forecasting.
5. To develop a web-based rainfall prediction and alert interface using the selected model.

## Future Scope

- Integration of additional meteorological variables.
- Evaluation across more geographical regions.
- Extended forecasting horizons.
- Improved uncertainty estimation and model interpretability.
- Validation of rainfall alert thresholds with local climate and disaster-management standards.
- Deployment and evaluation using continuously updated rainfall data.

## Repository

**GitHub Repository:**  
[Enhanced Multivariate Rainfall Forecasting with an Intelligent Rainfall Alert System](https://github.com/Solitaryseeker/Enhanced_Multivariate_Rainfall_Forecasting_with_an_Intelligent_Rainfall_Aler_-System)

**Web Application Source:**  
[Yemen Rainfall Prediction Application](https://github.com/Solitaryseeker/Enhanced_Multivariate_Rainfall_Forecasting_with_an_Intelligent_Rainfall_Aler_-System/tree/main/yemen-rainfall)

## Author

**Rohit Sahu**  
B.Tech – Computer Science and Engineering

GitHub: [Solitaryseeker](https://github.com/Solitaryseeker)

---

*This project is developed as an academic and research-oriented exploration of deep learning-based multivariate rainfall forecasting and intelligent rainfall alert systems.*
