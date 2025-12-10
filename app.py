

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Wine Quality Prediction App", layout="wide")

# ---------------------------------------------------
# BACKGROUND + STYLE
# ---------------------------------------------------
def set_background():
    bg_url = "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.55);
        z-index: -1;
    }}
    .glass {{
        background: rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        color: white !important;
    }}
    h1,h2,h3,h4,h5,h6,label,span,p {{
        color: #ffffff !important;
        font-weight: 600;
    }}
    .stButton>button {{
        background-color: #9b0000;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 16px;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_background()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
DATA_PATH = r"C:\Users\ADITHYA\OneDrive\Desktop\Wine Quality\winequalityN.csv"
df = pd.read_csv(DATA_PATH)
df["type"] = df["type"].astype("category")

st.markdown("<h1 class='glass'>🍷 Wine Quality Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------------------------------
# ENCODING + MODEL TRAINING
# ---------------------------------------------------
df_model = df.copy()
le = LabelEncoder()
df_model["type"] = le.fit_transform(df_model["type"])
df_model = df_model.dropna()

X = df_model.drop("quality", axis=1)
y = df_model["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].mean())

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
menu = st.sidebar.radio("📌 Navigate", ["Dataset", "Training", "Visualizations", "Prediction"])

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------
if menu == "Dataset":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# TRAINING METRICS
# ---------------------------------------------------
elif menu == "Training":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("⚙️ Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("📉 MSE", round(mse, 3))
    col2.metric("📈 R² Score", round(r2, 3))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# VISUALIZATIONS (TABS + SMALLER GRAPHS)
# ---------------------------------------------------
elif menu == "Visualizations":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Quality", "Alcohol vs Quality", "pH", "Sulphates vs Quality",
        "Volatile Acidity vs Quality", "Citric Acid"
    ])

    def small_plot():
        # Use smaller figure size and DPI
        return plt.figure(figsize=(3,2), dpi=100)

    with tab1:
        fig = small_plot()
        plt.hist(df["quality"], bins=10, color="crimson")
        plt.title("Quality Distribution")
        st.pyplot(fig)

    with tab2:
        fig = small_plot()
        plt.scatter(df["alcohol"], df["quality"], color="gold")
        plt.title("Alcohol vs Quality")
        plt.xlabel("Alcohol"); plt.ylabel("Quality")
        st.pyplot(fig)

    with tab3:
        fig = small_plot()
        plt.hist(df["pH"], bins=20, color="teal")
        plt.title("pH Distribution")
        st.pyplot(fig)

    with tab4:
        fig = small_plot()
        plt.scatter(df["sulphates"], df["quality"], color="purple")
        plt.title("Sulphates vs Quality")
        plt.xlabel("Sulphates"); plt.ylabel("Quality")
        st.pyplot(fig)

    with tab5:
        fig = small_plot()
        plt.scatter(df["volatile acidity"], df["quality"], color="orange")
        plt.title("Volatile Acidity vs Quality")
        plt.xlabel("Volatile Acidity"); plt.ylabel("Quality")
        st.pyplot(fig)

    with tab6:
        fig = small_plot()
        plt.hist(df["citric acid"], bins=20, color="green")
        plt.title("Citric Acid Distribution")
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# PREDICTION UI
# ---------------------------------------------------
elif menu == "Prediction":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🔮 Predict Wine Quality")

    input_values = {}
    for col in X.columns:
        if col == "type":
            input_values[col] = st.selectbox("Select Wine Type", ["red", "white"])
        else:
            input_values[col] = st.number_input(
                f"Enter {col}", float(df[col].min()), float(df[col].max())
            )

    if st.button("Predict Quality"):
        type_value = input_values["type"]
        type_numeric = 0 if type_value == "red" else 1
        input_values["type"] = type_numeric

        input_array = np.array(list(input_values.values())).reshape(1, -1)
        quality_pred = model.predict(input_array)[0]

        st.success(f"### 🍷 Predicted Wine Quality: **{round(quality_pred, 2)}**")

    st.markdown("</div>", unsafe_allow_html=True)


