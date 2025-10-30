# ModelFitter

**ModelFitter** is a Python-based interactive application with a UI (using :contentReference[oaicite:1]{index=1}) that predicts whether an ML model will **overfit**, **underfit** or be **well-fit (normal fit)** given dataset characteristics and certain hyperparameters.

## Features

- Calculate meta-features of the dataset (e.g., number of informative features, class separability) via `meta_calculation.py`.  
- Accept user input (model type, hyperparameters, dataset features) and output a prediction of fit-quality.  
- UI built with Streamlit for easy interaction and visualization.  
- Includes sample datasets and pre-trained model for quick start.

## Why Use It

When building machine learning models, you often wonder:  
> “Is my model going to overfit or underfit given these hyperparameters and dataset properties?”  
This tool provides a lightweight, interactive way to **estimate model-fit behaviour** before or during training, helping you to:  
- Adjust hyperparameters early.  
- Understand dataset complexity and how it might impact fit.  
- Quickly prototype and test different scenarios.

## Installation & Setup

1. Clone the repository:  
   ```bash
   git clone https://github.com/abhijay029/ModelFitter.git
   cd ModelFitter
2. Create a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Dependancies
   ```bash
   pip install -r requirements.txt
4. Run Locally
   ```bash
   streamlit run app.py
## Project Structure
ModelFitter/
│
├─ app.py                 # the Streamlit UI entrypoint  
├─ meta_calculation.py    # functions to compute meta-features  
├─ data_generator.py      # helper to simulate/generate datasets  
├─ requirements.txt       # list of dependencies for deployment  
├─ test_dataset/          # sample datasets  
├─ test_cases/            # example scenarios / notebooks  
├─ XGBoost_model_unscaled.pkl  # saved pre-trained model (example)  
└─ …
