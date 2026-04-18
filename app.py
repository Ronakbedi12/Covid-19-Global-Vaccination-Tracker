import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time


st.set_page_config(page_title="COVID-19 Tracker", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00FFAA;
}
.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("vaccination_model.pkl")

model = load_model()

st.title("COVID-19 Global Vaccination Tracker")
st.markdown("Predict vaccination rate and get safety recommendations")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Charts" , "COVID Info", "Symptoms", "Precautions", "Global Vaccination"]
)

if menu == "Prediction":
    st.header("Vaccination Tracker")

    col1, col2 = st.columns(2)

    with col1:
        total_cases = st.number_input("Total Cases", value=10000, key="tc")
        total_deaths = st.number_input("Total Deaths", value=500, key="td")

    with col2:
        daily_cases = st.number_input("Daily Cases", value=100, key="dc")
        daily_deaths = st.number_input("Daily Deaths", value=5, key="dd")

    col3, col4 = st.columns(2)

    with col3:
        population = st.number_input("Population", value=1000000, key="pop")
        infection_rate = st.number_input("Infection Rate(%)", value=5, key="ir")

    with col4:
        year = st.number_input("Year", value=2021)
        month = st.number_input("Month", value=6)

    vaccination_gap = st.number_input("Vaccination Gap" , value=10000)

    if st.button("Predict" , key="predict_button_1"):
        with st.spinner("Analyzing data... Please wait "):
            time.sleep(1)

            cases_per_death = (total_cases / (total_deaths + 1))*100
            input_data = pd.DataFrame({
                "Total Cases": [total_cases],
                "Daily Cases": [daily_cases],
                "Total Deaths": [total_deaths],
                "Daily Deaths": [daily_deaths],
                "Population": [population],
                "Infection Rate(%)": [infection_rate],
                "Year": [year],
                "Month": [month],
                "Cases_per_Death": [cases_per_death],
                "Vaccination_Gap": [vaccination_gap]
        })

        prediction = model.predict(input_data)[0]

        st.success(f"Tracked Vaccination Rate: {prediction*10:.2f}%")

        if prediction < 30:
            st.warning("Low vaccination rate. Increase awareness and vaccination drives.")
        elif prediction < 60:
            st.info("Moderate vaccination. Continue vaccination efforts.")
        else:
            st.success("High vaccination. Population is relatively safer.")

elif menu == "Charts":

    st.header("Data Visualization")

    @st.cache_data
    def load_data():
        return pd.read_csv("Cleaned_Data.csv")

    df = load_data()
    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    countries = sorted(df["country"].dropna().unique())
    selected_country = st.selectbox("Select Country", countries)
    df_country = df[df["country"] == selected_country]

    df_latest = df.sort_values("date").groupby("country").tail(1)
    
    st.subheader(" Global COVID-19 Cases Map ")
    fig_map = px.choropleth(
        df_latest,
        locations="country",
        locationmode="country names",
        color="total cases",
        hover_name="country",
        color_continuous_scale="reds",
        title="Global COVID-19 Spread"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader(f" Trend for {selected_country}")

    col1, col2 = st.columns(2)

    fig1 = px.line(
        df_country,
        x="date",
        y="total cases",
        title="Total Cases Over Time"
    )

    fig2 = px.line(
        df_country,
        x="date",
        y="total deaths",
        title="Total Deaths Over Time"
    )

    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Countries (Latest Data)")

    df_top = df_latest.sort_values("total cases", ascending=False).head(10)

    fig3 = px.bar(
        df_top,
        x="country",  
        y="total cases",
        title="Top 10 Countries by Total Cases",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Cases Distribution")

    fig4 = px.pie(
        df_top,
        names="country",   
        values="total cases",
        title="Cases Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)

elif menu == "COVID Info":
    st.header("About COVID-19")
    st.write("""
    Coronavirus disease 2019 (COVID-19) is a contagious disease caused by the coronavirus SARS-CoV-2. Starting in January 2020, the disease spread worldwide, resulting in the COVID-19 pandemic. In March 2020, the World Health Organization declared COVID-19 a global health emergency; they declared the end of the emergency in May 2023.

The symptoms of COVID‑19 can vary but often include fever, fatigue, cough, breathing difficulties, loss of smell, and loss of taste. Symptoms may begin one to 14 days after exposure to the virus. At least a third of people who are infected do not develop noticeable symptoms. Of those who develop symptoms noticeable enough to be classified as patients, most (81%) develop mild to moderate symptoms (up to mild pneumonia), while 14% develop severe symptoms (dyspnea, hypoxia, or more than 50% lung involvement on imaging), and 5% develop critical symptoms (respiratory failure, shock, or multiorgan dysfunction).Older people have a higher risk of developing severe symptoms and dying. Some people experience persistent symptoms (long COVID), for months or years after infection, including fatigue, cognitive issues and shortness of breath. Damage to organs has been observed in a subset.

COVID‑19 transmission occurs when infectious particles are breathed in or come into contact with the eyes, nose, or mouth. The risk is highest when people are in close proximity, but small airborne particles containing the virus can remain suspended in the air and travel over longer distances, particularly indoors. Transmission can also occur when people touch their eyes, nose, or mouth after touching surfaces or objects that have been contaminated by the virus. People remain contagious for up to 20 days and can spread the virus even if they do not develop symptoms.

There are two common tests to detect COVID. Antigen tests (also called rapid lateral flow tests) can be used at home. A positive test indicates an active infection. However, negative test results are not always accurate, especially early or late in the infection. Health care providers can perform a more accurate PCR test, which is typically analysed in a laboratory.

Several COVID-19 vaccines have been approved and distributed in various countries, many of which have initiated mass vaccination campaigns. Other preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, use of face masks or coverings in public, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. While drugs have been developed to inhibit the virus, the primary treatment is still symptomatic, managing the disease through supportive care, isolation, and experimental measures.

The first known case was identified in Wuhan, China, in December 2019.Most scientists believe that the SARS-CoV-2 virus entered into human populations through natural zoonosis, similar to the SARS-CoV-1 and MERS-CoV outbreaks, and consistent with other pandemics in human history. Social and environmental factors including climate change, natural ecosystem destruction and wildlife trade increased the likelihood of such zoonotic spillover.[
    """)

elif menu == "Symptoms":
    st.header("Symptoms of COVID-19")

    st.write("""
    - Fever
    - Dry cough
    - Tiredness
    - Loss of taste or smell
    - Difficulty breathing
             
    The symptoms of COVID-19 are variable depending on the type of variant contracted, ranging from mild symptoms to a potentially fatal illness. Common symptoms include coughing, fever, loss of smell and taste, with less common ones including headaches, nasal congestion and runny nose, muscle pain, sore throat, diarrhea, eye irritation, and toes swelling or turning purple, and in moderate to severe cases, breathing difficulties.People with the COVID-19 infection may have different symptoms, and their symptoms may change over time.
    """)

elif menu == "Precautions":
    st.header("Precautions")

    st.write("""
    - Wear masks 😷
    - Wash hands regularly 🧼
    - Maintain social distance 📏
    - Get vaccinated 💉
             
    Preventive measures to reduce the chances of infection include getting vaccinated, staying at home, wearing a mask in public, avoiding crowded places, keeping distance from others, ventilating indoor spaces, managing potential exposure durations, washing hands with soap and water often and for at least 20 seconds, practising good respiratory hygiene, and avoiding touching the eyes, nose, or mouth with unwashed hands.
    """)

elif menu == "Global Vaccination":
    st.header("Global Vaccination Info")

    st.write("""
    Major vaccines include the Pfizer–BioNTech mRNA vaccine, Moderna mRNA vaccine, and the Novavax protein subunit vaccine.With the emergence of new SARS-CoV-2 variants, the original vaccines—particularly Pfizer–BioNTech and Moderna vaccines—have been updated. These "variant-adapted" vaccines are offered as booster doses.[210] The immunity from the vaccines also wanes over time, requiring people to get boosters to maintain protection.

    Common side effects of COVID‑19 vaccines include soreness, fatigue, headache, myalgia (muscle pain), and arthralgia (joint pain), which resolve without medical treatment within a few days. COVID‑19 vaccination is safe for people who are pregnant or are breastfeeding.

    The COVID‑19 vaccines are widely credited for their role in reducing the spread of COVID‑19 and reducing the severity and death caused by COVID‑19. Many countries implemented phased distribution plans that prioritized those at highest risk of complications, such as the elderly, and those at high risk of exposure, such as healthcare workers.By December 2020, more than 10 billion vaccine doses had been preordered,with about half of the doses purchased by high-income countries comprising 14% of the world's population.As of August 2024, over 13 billion doses of COVID‑19 vaccines have been administered worldwide.
    """)