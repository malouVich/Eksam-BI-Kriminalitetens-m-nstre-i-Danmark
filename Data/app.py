import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# --------------------------
# 1. Load clean dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clean_crime_data.csv")

    # Keep only valid categories
    valid_status = ["I alt","Selvstændige","Lønmodtagere","Arbejdsløse","Studerende",
                    "Pensionister mv.","Øvrige personer","Uoplyst"]
    df = df[df["Category3"].isin(valid_status)]
    df = df[df["Category1"].isin(["Mænd","Kvinder"])]

    return df

df1 = load_data()
years = [str(y) for y in range(2015, 2024)]
year_cols = [y for y in years if y in df1.columns]

# --------------------------
# Sidebar filters
# --------------------------
st.sidebar.header("Indstillinger")
status_choice = st.sidebar.selectbox(
    "Vælg socioøkonomisk status:",
    ["Alle"] + df1["Category3"].unique().tolist()
)
gender_choice = st.sidebar.radio(
    "Vælg køn:",
    ["Alle"] + df1["Category1"].unique().tolist()
)

# Filter dataset based on sidebar
filtered_df = df1.copy()
if status_choice != "Alle":
    filtered_df = filtered_df[filtered_df["Category3"] == status_choice]
if gender_choice != "Alle":
    filtered_df = filtered_df[filtered_df["Category1"] == gender_choice]

# --------------------------
# Tabs
# --------------------------
st.title("Kriminalitetens Mønstre i Danmark (2015-2023)")
tab1, tab2, tab3 = st.tabs(["📊 Dataanalyse", "🤖 ML Forudsigelser", "🟢 Klyngedannelse"])

# --------------------------
# TAB 1: Data Analysis
# --------------------------
with tab1:
    st.header("Dataanalyse")

    fig, ax = plt.subplots(figsize=(10,6))
    totals = filtered_df[year_cols].sum()
    ax.plot(year_cols, totals, marker="o", label="Total")
    ax.set_title(f"Udvikling i kriminalitet ({status_choice}, {gender_choice})")
    ax.set_xlabel("År")
    ax.set_ylabel("Antal personer")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"""
    **Observation:**  
    I den valgte kategori *{status_choice}* ses udviklingen fordelt på *{gender_choice}* fra 2015 til 2023.
    """)

# --------------------------
# TAB 2: ML Predictions
# --------------------------
with tab2:
    st.header("ML Forudsigelser")
    st.markdown("""
    Simpelt eksempel med **lineær regression** for at forudsige kriminalitetstal fremad.
    """)

    df_total = filtered_df[year_cols].sum().reset_index()
    df_total.columns = ["Year", "Value"]
    df_total["Year"] = df_total["Year"].astype(int)

    X = df_total[["Year"]]
    y = df_total["Value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"**Model performance:** R² = {r2_score(y_test, y_pred):.2f}, MSE = {mean_squared_error(y_test, y_pred):.2f}")

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

# --------------------------
# TAB 3: Clustering
# --------------------------
with tab3:
    st.header("Klyngedannelse af kriminalitetsdata")
    st.markdown("Vælg årstal til clustering:")

    selected_features = st.multiselect("Vælg år", options=year_cols, default=year_cols[:2])

    if len(selected_features) >= 2:
        X_selected = filtered_df[selected_features].fillna(filtered_df[selected_features].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        n_clusters = st.slider("Vælg antal klynger", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        filtered_df['ClusterNum'] = cluster_labels

        # Assign meaningful cluster names
        filtered_df['TotalCrime'] = filtered_df[selected_features].sum(axis=1)
        cluster_totals = filtered_df.groupby('ClusterNum')['TotalCrime'].mean().sort_values()
        crime_labels = ["Lav kriminalitet", "Middel kriminalitet", "Høj kriminalitet"]
        if n_clusters != 3:
            crime_labels = [f"Klynge {i+1}" for i in range(n_clusters)]
        cluster_name_map = {cluster: crime_labels[i] for i, cluster in enumerate(cluster_totals.index)}
        filtered_df['Cluster'] = filtered_df['ClusterNum'].map(cluster_name_map)

        x_axis = st.selectbox("X-akse", selected_features, index=0)
        y_axis = st.selectbox("Y-akse", selected_features, index=1)

        plt.figure(figsize=(8,6))
        sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue='Cluster', palette='tab10', s=100)
        plt.title("Klyngedannelse af kriminalitet")
        st.pyplot(plt)

        st.markdown(f"**Observation:** {n_clusters} klynger opdelt baseret på valgte år med meningsfulde navne.")
    else:
        st.warning("Vælg mindst 2 år for at udføre clustering.")
