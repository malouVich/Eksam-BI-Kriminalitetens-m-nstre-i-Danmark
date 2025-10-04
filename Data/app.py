import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# --------------------------
# 1. Data Loading
# --------------------------
@st.cache_data
def load_data():
    df1 = pd.read_excel("Skyldige personer efter køn, alder, socioøkoknomisk status og tid(2015-2023).xlsx", header=2)
    df1['Unnamed: 0'] = df1['Unnamed: 0'].ffill()
    df1['Unnamed: 1'] = df1['Unnamed: 1'].ffill()
    return df1

df1 = load_data()
years = ["2015","2016","2017","2018","2019","2020","2021","2022","2023"]

# --------------------------
# 2. Tabs
# --------------------------
st.title("Kriminalitetens Mønstre i Danmark (2015-2023)")

tab1, tab2, tab3 = st.tabs(["📊 Dataanalyse", "🤖 ML Forudsigelser", "🟢 Klyngedannelse"])

# --------------------------
# TAB 1: Descriptive analysis
# --------------------------
with tab1:
    st.header("Dataanalyse")

    st.sidebar.header("Indstillinger")
    status_choice = st.sidebar.selectbox("Vælg socioøkonomisk status:", df1["Unnamed: 2"].dropna().unique())
    gender_choice = st.sidebar.radio("Vælg køn:", ["Alle", "Mænd", "Kvinder"])

    subset = df1[df1["Unnamed: 2"] == status_choice]

    fig, ax = plt.subplots(figsize=(10,6))
    if gender_choice == "Alle":
        for gender in ["Mænd", "Kvinder"]:
            df_gender = subset[subset["Unnamed: 0"] == gender]
            totals = df_gender[years].sum()
            ax.plot(years, totals, marker="o", label=gender)
    else:
        df_gender = subset[subset["Unnamed: 0"] == gender_choice]
        totals = df_gender[years].sum()
        ax.plot(years, totals, marker="o", label=gender_choice)

    ax.set_title(f"Udvikling i kriminalitet ({status_choice})")
    ax.set_ylabel("Antal personer")
    ax.set_xlabel("År")
    ax.legend()

    st.pyplot(fig)

    st.markdown(f"""
    **Observation:**  
    I den valgte kategori *{status_choice}* ses udviklingen fordelt på {gender_choice.lower()} fra 2015 til 2023.  
    """)

# --------------------------
# TAB 2: Machine Learning Predictions
# --------------------------
with tab2:
    st.header("ML Forudsigelser")

    st.markdown("""
    Her demonstreres et simpelt eksempel på brug af **lineær regression** for at forudsige kriminalitetstal
    i de kommende år baseret på historiske data.
    """)

    # Example: Use totals per year (all statuses + genders combined)
    df_total = df1[years].sum().reset_index()
    df_total.columns = ["Year", "Value"]
    df_total["Year"] = df_total["Year"].astype(int)

    X = df_total[["Year"]]
    y = df_total["Value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Model performance:** R² = {r2:.2f}, MSE = {mse:.2f}")

    # Predict future years
    future_years = pd.DataFrame({"Year": [2024, 2025, 2026, 2027]})
    future_preds = model.predict(future_years)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(X, y, label="Historiske data", marker="o")
    ax.plot(X_test, y_pred, label="Test forudsigelser", linestyle="--", color="red")
    ax.plot(future_years["Year"], future_preds, label="Fremtidige prognoser", marker="x", color="green")
    ax.set_title("Kriminalitet: historiske data og forudsigelser")
    ax.set_xlabel("År")
    ax.set_ylabel("Antal personer")
    ax.legend()

    st.pyplot(fig)

    st.markdown("👉 Dette er blot et eksempel. I Sprint 3 kan I erstatte modellen med mere avancerede ML-metoder (klassifikation, klyngedannelse, neural netværk osv.).")


# --------------------------
# TAB 3: Clustering
# --------------------------
with tab3:
    st.header("Klyngedannelse af kriminalitetsdata")

    numeric_features = years  # Only numeric year columns for clustering
    st.markdown("Vælg årstal til clustering:")
    selected_features = st.multiselect("Vælg år", options=numeric_features, default=numeric_features[:2])

    if len(selected_features) >= 2:
        # Handle missing values by filling with mean
        X_selected = df1[selected_features].fillna(df1[selected_features].mean())

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Number of clusters
        n_clusters = st.slider("Vælg antal klynger", 2, 10, 3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df1['Cluster'] = kmeans.fit_predict(X_scaled)

        # Select axes for plotting
        x_axis = st.selectbox("X-akse", selected_features, index=0)
        y_axis = st.selectbox("Y-akse", selected_features, index=1)

        # Plot clusters
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df1, x=x_axis, y=y_axis, hue='Cluster', palette='tab10', s=100)
        plt.title("Klyngedannelse af kriminalitet")
        st.pyplot(plt)

        st.markdown(f"**Observation:** {n_clusters} klynger opdelt baseret på valgte år.")
    else:
        st.warning("Vælg mindst 2 år for at udføre clustering.")
