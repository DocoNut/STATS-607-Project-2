# Multi Kernel Density Estimation (MKDE)

This project introduces a new **density estimator** based on the traditional Kernel Density Estimator (KDE).  
Our MKDE method is **unbiased** and often more powerful than the original KDE.

---

## 📌 Features

### Models
- **`data_preparation.py`** – Generate samples and PDFs for F, Normal, Beta, and Bimodal distributions.  
- **`mulkde_coef.py`** – Compute coefficients for MKDE.  
- **`mulkde.py`** – Implementation of the MKDE function.  
- **`other_kde.py`** – Alternative density estimators for comparison.  
- **`data_preprocessing.py`** – Preprocess raw data.  
- **`mkde_comparison.py`** – Compare MKDE with other density estimators on synthetic models.  
- **`mkde_simu.py`** – Compare MKDE with other density estimators on the *Old Faithful* dataset.  

### Tests
- **`tests/data_test.py`** – Unit tests for processed data.  
- **`tests/function_test.py`** – Unit tests for core model functions.  

### Main Script
- **`run_analysis.py`** – End-to-end pipeline:
  1. Preprocess data  
  2. Run comparisons  
  3. Test on real data  

---

## ⚙️ Installation

We recommend running the project in a **virtual environment**.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate     # on macOS/Linux
venv\Scripts\activate        # on Windows
## 📦 Dependencies
```

The project requires the following Python packages (see `requirements.txt` for full details):

- numpy  
- scipy  
- matplotlib  
- pandas  
- pytest  

Install them with:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

> **Important**: Run all scripts from the **project root directory**.

Run the full pipeline (data preprocessing, comparison, and real data test):

```bash
python3 -m run_analysis
```

Run specific modules individually:

```bash
# Data preprocessing
python3 -m models.data_preprocessing
# Comparison on synthetic models
python3 -m models.mkde_comparison
# Simulation on real data (Old Faithful dataset)
python3 -m models.mkde_simu
```

Run tests:
```bash
python3 -m tests.data_test
python3 -m tests.function_test
```

Or run with Pytest:
```bash
pytest tests/
```
