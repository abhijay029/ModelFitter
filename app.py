import streamlit as st
from meta_calculation import get_class_separability, get_n_informative
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def meta_extraction():
    container = st.session_state["meta_output"]
    dataset = st.session_state["train_data"]

    if "target_col" not in st.session_state or st.session_state["target_col"] == "":
        container.warning("Please provide the name of the target column.")
        return
    
    df = pd.read_csv(dataset)
    try:
        y = df[st.session_state["target_col"]]
    except KeyError:
        container.warning(f"Target column '{st.session_state['target_col']}' not found in the dataset. Please check the column name.")
        return
    
    X = df.drop(columns = [st.session_state["target_col"]])
    
    n_classes = y.nunique()
    n_features = df.shape[1] - 1
    n_samples = df.shape[0]

    n_informative = get_n_informative(X, y)
    class_sep = get_class_separability(X, y)
    
    st.session_state["n_classes"] = n_classes
    st.session_state["n_features"] = n_features
    st.session_state["n_samples"] = n_samples
    st.session_state["n_informative"] = n_informative
    st.session_state["class_sep"] = class_sep

    
    
    container.subheader("Dataset information")
    container.write(f"Number of samples: {n_samples}")
    container.write(f"Number of features: {n_features}")
    container.write(f"Number of classes: {n_classes}")
    container.write(f"Number of informative features: {n_informative}")
    container.write(f"Class separability: {class_sep}")

def predict_fit():
    container = st.session_state["prediction_output"]
    if st.session_state.get("n_classes") is None or st.session_state.get("n_features") is None or st.session_state.get("n_samples") is None or st.session_state.get("n_informative") is None or st.session_state.get("class_sep") is None:
        container.warning("Please extract meta-features first.")
        return
    X = {
        "max_depth": [st.session_state["max_depth"]],
        "min_samples_split": [st.session_state["min_samples_split"]],
        "min_samples_leaf": [st.session_state["min_samples_leaf"]],
        "n_samples": [st.session_state["n_samples"]],
        "n_features": [st.session_state["n_features"]],
        "n_informative": [st.session_state["n_informative"]],
        "class_sep": [st.session_state["class_sep"]],
        "n_classes": [st.session_state["n_classes"]],
        "criterion_entropy": [1 if st.session_state["criterion"] == "entropy" else 0],
        "criterion_gini": [1 if st.session_state["criterion"] == "gini" else 0],
        "criterion_log_loss": [1 if st.session_state["criterion"] == "log_loss" else 0]
        }
    df_input = pd.DataFrame(X)
    df_input = pd.get_dummies(df_input)

    model = pickle.load(open("XGBoost_model_unscaled.pkl", "rb"))
    prediction = model.predict(df_input)
    outcome = "Underfit" if prediction[0] == 0 else ("Normal Fit" if prediction[0] == 1 else "Overfit")
    container.subheader("Prediction Result")
    container.write(f"The model is predicted to **{outcome}**")

st.title("Decision Tree Classifier")
st.subheader("Please provide the parameters of your Decision Tree Classifier")

st.number_input(label = "max_depth", key="max_depth", value = -1, step = 1)
st.caption("Note: Use -1 if max_depth is not specified or None specified in your model.")

st.number_input(label = "min_samples_split", key="min_samples_split", value = 2, step = 1, min_value=2)
st.caption("Note: Use 2 if min_samples_split is not specified or None specified in your model.")

st.number_input(label = "min_samples_leaf", key="min_samples_leaf", value = 1, step = 1, min_value=1)
st.caption("Note: Use 1 if min_samples_leaf is not specified or None specified in your model.")

st.selectbox(label = "criterion", key="criterion", options = ["gini", "entropy", "log_loss"])
st.caption("Note: Keep this gini if not specified in your model.")


st.subheader("Please give the training dataset")
st.file_uploader(label = "Upload CSV", type=["csv"], key="train_data")
st.text_input("Name of target column", key="target_col", value="target")

st.button("Extract Meta-features", on_click = meta_extraction)
st.session_state["meta_output"] = st.container()

st.button("Predict", on_click = predict_fit)
st.session_state["prediction_output"] = st.container()
