import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import pickle

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

    st.markdown("""
### Forklaring af dataanalysen

Denne fane viser udviklingen i kriminalitet fordelt på køn og socioøkonomisk status i perioden **2015–2023**.  
Ved at vælge en bestemt socioøkonomisk status og køn i sidebaren, kan du se hvordan kriminaliteten har ændret sig over tid.

**Formål:**  
At identificere hvilke grupper, der har haft stigende eller faldende kriminalitetstendenser, og hvor der kan være behov for forebyggelse.

**Fortolkning af grafen:**
- **X-akse:** År (2015–2023)  
- **Y-akse:** Antal personer dømt for kriminalitet  
- **Farver:** Mænd og kvinder (eller begge kombineret)
- **Linjer:** Udvikling over tid for den valgte socioøkonomiske gruppe  

**Eksempel:**  
Hvis man vælger *arbejdsløse mænd*, kan man observere, om kriminaliteten er steget eller faldet i denne gruppe gennem årene.  

**Praktisk anvendelse:**  
Disse observationer kan bruges til at støtte **sociale initiativer, uddannelsesindsatser** eller **lokale forebyggelsesstrategier**.
""")

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
# TAB 2: ML Predictions with multiple features
# --------------------------
with tab2:
    st.header("ML Forudsigelser")
    st.markdown("""
    Simpelt eksempel med **lineær regression** for at forudsige kriminalitetstal fremad.
    """)

    # Summer alle værdier pr. år
    df_total = filtered_df[year_cols].sum().reset_index()
    df_total.columns = ["Year", "Value"]
    df_total["Year"] = df_total["Year"].astype(int)

    # Features og target
    X = df_total[["Year"]]
    y = df_total["Value"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Træn model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Beregn model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Model performance:** R² = {r2:.2f}, MSE = {mse:.2f}")

    # Fremtidige år og forudsigelser
    future_years = pd.DataFrame({"Year": [2024, 2025, 2026, 2027]})
    future_preds = model.predict(future_years)

    # Plot historiske data, test og fremtidige prognoser
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(X, y, label="Historiske data", marker="o")
    ax.plot(X_test, y_pred, label="Test forudsigelser", linestyle="--", color="red")
    ax.plot(future_years["Year"], future_preds, label="Fremtidige prognoser", marker="x", color="green")
    ax.set_title("Kriminalitet: historiske data og forudsigelser")
    ax.set_xlabel("År")
    ax.set_ylabel("Antal personer")
    ax.legend()
    st.pyplot(fig)

    # Forklaring af forudsigelserne
    st.markdown(f"""
**Forklaring af forudsigelserne:**  

Vi bruger en **lineær regressionsmodel** til at forudsige kriminalitetstal i Danmark for de kommende år baseret på historiske data (2015-2023). Modellen estimerer antallet af personer, der vil blive fundet skyldige, og tager udgangspunkt i den observerede udvikling i de seneste år.  

- **Historiske data:** viser den faktiske udvikling i kriminalitetstal fra 2015 til 2023.  
- **Testforudsigelser:** viser, hvor godt modellen passer på de seneste kendte data.  
- **Fremtidige prognoser:** viser forventede værdier for 2024-2027 baseret på trends.  

**Modelpræstation:**  
- R² = {r2:.2f} → modellen forklarer {r2*100:.0f}% af variationen i data.  
- MSE = {mse:.2f} → angiver den gennemsnitlige kvadratiske fejl mellem modelens forudsigelser og de faktiske tal.  

**Bemærk:** Lineær regression antager en konstant trend. Pludselige ændringer (fx nye love eller samfundsmæssige begivenheder) kan påvirke nøjagtigheden. Prognoserne bliver mere usikre, jo længere ud i fremtiden de gælder.  

**Praktisk anvendelse:**  
Disse forudsigelser kan hjælpe kommuner og politimyndigheder med at planlægge ressourcer og målrette forebyggende indsats.
""")

# --------------------------
# TAB 3: Clustering
# --------------------------
with tab3:
    st.header("Klyngedannelse af kriminalitetsdata")
    st.markdown("""
### Forklaring af klyngedannelsen

I denne fane anvendes **K-Means klyngedannelse**, en metode fra *unsupervised machine learning*, til at finde mønstre i kriminalitetsdataene.

**Formål:**  
At gruppere lignende observationer sammen ud fra kriminalitetsniveau over udvalgte år.

**Sådan fungerer det:**  
- Data standardiseres (gennemsnit = 0, standardafvigelse = 1) for at gøre årstal sammenlignelige.  
- Brugeren vælger antallet af klynger (2–10).  
- Algoritmen fordeler observationerne i grupper, så variationen inden for hver gruppe minimeres.

**Fortolkning af grafen:**
- Hver prik repræsenterer en observation (fx en socioøkonomisk kategori).  
- **Farverne** viser, hvilken klynge observationen tilhører.  
- **X-akse og Y-akse:** De valgte årstal, som bruges til at sammenligne udviklingen.

**Eksempel:**  
Hvis to kategorier (fx “Arbejdsløse mænd” og “Ikke-arbejdende unge kvinder”) ligger tæt i diagrammet, betyder det, at deres kriminalitetsmønstre over tid ligner hinanden.

**Praktisk anvendelse:**  
Klyngedannelse kan hjælpe beslutningstagere med at identificere:
- Hvilke grupper der opfører sig ens over tid  
- Hvor der kan sættes målrettede sociale eller forebyggende tiltag ind  
""")
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
