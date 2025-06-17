import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import joblib
import math

import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
import shap 
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler



st.set_page_config(
    page_title="Aortic Valve Treatment Tool",
    layout="wide",  
    initial_sidebar_state="expanded"
)






if 'random_patient' not in st.session_state:
    st.session_state.random_patient = None


user_input = {
    'age': None,
    'sex': None,
    'height': None,
    'weight': None,
    'lvef': None,
    'priorhf': None,
    'dm': None,
    'hypertension': None,
    'dialysis': None,
    'copd': None,
    'cvd': None,
    'shock': None,
    'priormi': None,
    'priorpad': None,
    'preproccreat': None,
    'hgb': None,
    'plateletct': None,
    'albumin': None,
    'inr': None,
    'nyha': None
}


def generate_random_patient():
    st.session_state.random_patient = {
        'age': random.randint(60, 90),
        'sex': random.choice(["Male", "Female"]),
        'height': round(random.uniform(140.0, 210.0), 1),
        'weight': round(random.uniform(36.0, 200.0), 1),
        'lvef': random.randint(5, 90),
        'priorhf': random.choice(["Yes", "No"]),
        'dm': random.choice(["Yes", "No"]),
        'hypertension': random.choice(["Yes", "No"]),
        'dialysis': random.choice(["Yes", "No"]),
        'copd': random.choice(["Yes", "No"]),
        'cvd': random.choice(["Yes", "No"]),
        'shock': random.choice(["Yes", "No"]),
        'priormi': random.choice(["Yes", "No"]),
        'priorpad': random.choice(["Yes", "No"]),
        'preproccreat': round(random.uniform(0.4, 5.0), 2),
        'hgb': round(random.uniform(5.0, 20.0), 1),
        'plateletct': random.randint(50000, 500000),
        'albumin': round(random.uniform(2.0, 5.2), 1),
        'inr': round(random.uniform(0.8, 1.6), 1),
        'nyha': random.choice(["I", "II", "III", "IV"])
    }





# ===========================
# Sidebar Input Form
# ===========================

with st.sidebar:
    user_input['age'] = st.slider(
        "Age:", 18, 100,
        st.session_state.random_patient['age'] if st.session_state.random_patient else 18
    )
    user_input['sex'] = st.selectbox(
        "Gender:", ["Male", "Female"],
        index=["Male", "Female"].index(st.session_state.random_patient['sex']) if st.session_state.random_patient else 0
    )
    user_input['height'] = st.slider(
        "Height (cm):", 140.0, 210.0,
        st.session_state.random_patient['height'] if st.session_state.random_patient else 140.0, step=0.1
    )
    user_input['weight'] = st.slider(
        "Weight (kg):", 36.0, 200.0,
        st.session_state.random_patient['weight'] if st.session_state.random_patient else 36.0, step=0.1
    )
    user_input['lvef'] = st.slider(
        "Left Ventricular Ejection Fraction (%):", 5, 90,
        st.session_state.random_patient['lvef'] if st.session_state.random_patient else 5
    )
    user_input['priorhf'] = st.radio(
        "Prior Heart Failure:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['priorhf']) if st.session_state.random_patient else 0
    )
    user_input['dm'] = st.radio(
        "Diabetes:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['dm']) if st.session_state.random_patient else 0
    )
    user_input['hypertension'] = st.radio(
        "Hypertension:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['hypertension']) if st.session_state.random_patient else 0
    )
    user_input['dialysis'] = st.radio(
        "Currently on Dialysis:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['dialysis']) if st.session_state.random_patient else 0
    )
    user_input['copd'] = st.radio(
        "COPD:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['copd']) if st.session_state.random_patient else 0
    )
    user_input['cvd'] = st.radio(
        "Cerebrovascular Disease:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['cvd']) if st.session_state.random_patient else 0
    )
    user_input['shock'] = st.radio(
        "Cardiogenic Shock:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['shock']) if st.session_state.random_patient else 0
    )
    user_input['priormi'] = st.radio(
        "Prior Myocardial Infarction:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['priormi']) if st.session_state.random_patient else 0
    )
    user_input['priorpad'] = st.radio(
        "Prior Peripheral Arterial Disease:", ["No", "Yes"], horizontal=True,
        index=["No", "Yes"].index(st.session_state.random_patient['priorpad']) if st.session_state.random_patient else 0
    )
    user_input['preproccreat'] = st.slider(
        "Creatinine (mg/dL):", 0.4, 5.0,
        st.session_state.random_patient['preproccreat'] if st.session_state.random_patient else 0.4, step=0.01
    )
    user_input['hgb'] = st.slider(
        "Hemoglobin (g/dL):", 5.0, 20.0,
        st.session_state.random_patient['hgb'] if st.session_state.random_patient else 5.0, step=0.1
    )
    user_input['plateletct'] = st.slider(
        "Platelet Count (μL):", 50000, 500000,
        st.session_state.random_patient['plateletct'] if st.session_state.random_patient else 50000, step=1000
    )
    user_input['albumin'] = st.slider(
        "Albumin (g/dL):", 2.0, 5.2,
        st.session_state.random_patient['albumin'] if st.session_state.random_patient else 2.0, step=0.1
    )
    user_input['inr'] = st.slider(
        "International Normalized Ratio (INR):", 0.8, 1.6,
        st.session_state.random_patient['inr'] if st.session_state.random_patient else 0.8, step=0.1
    )
    user_input['nyha'] = st.selectbox(
        "NYHA Class:", ["I", "II", "III", "IV"],
        index=["I", "II", "III", "IV"].index(st.session_state.random_patient['nyha']) if st.session_state.random_patient else 0
    )


    st.divider()

    col1, col2 = st.columns([1.5, 1]) 
    with col1:
        random_patient_btn = st.button("Random Patient", on_click=generate_random_patient)
        
    with col2:
        submit_btn = st.button("Submit")

    if random_patient_btn:
        st.success("Random patient generated successfully!")

    if submit_btn:
        if not all(user_input.values()):
            st.warning("Please fill in all fields.")




# ===========================
# Intro and Overview
# ===========================

st.header("Aortic Valve Treatment Decision Support Tool")
st.markdown("Welcome to the Aortic Valve Treatment Explorer — an interactive tool designed to help visualize and compare outcomes for patients undergoing **Transcatheter Aortic Valve Replacement (TAVR)** or **Surgical Aortic Valve Replacement (SAVR)**.")
st.markdown("This tool is built using real-world clinical data from Northern New England region, enabling personalized insight into how your profile compares to thousands of patients treated in the area.")


#st.markdown("<br><br>", unsafe_allow_html=True)  # one line
st.divider()


with st.container(border=False):
    # display metrics
    m_col1, m_col2 = st.columns([1,1])
    m_col1.metric(label="Participating Hospitals", value="5")
    m_col2.metric(label="Data Collection Period", value="2015 - 2023")
    m_col1.metric(label="Total Patients", value="> 5,000")
    m_col2.metric(label="TAVR / SAVR Ratio", value="40% / 60%")

    st.markdown("<br>", unsafe_allow_html=True)  # one line

    # display overall umap
    st.subheader("UMAP Visualization of Overall Patient Profiles")
    st.markdown("The interactive UMAP plot below visualizes the distribution of patient profiles in the dataset. Each point represents a patient, colored by their treatment type (TAVR or SAVR).")


    df_plot = pd.read_csv('data/umap_embedding.csv')

    fig = px.scatter(
        df_plot,
        x="UMAP1",
        y="UMAP2",
        color="treatment",
        labels={"treatment": "Treatment"},
        opacity=0.6
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        dragmode=False,
        height=600,
        legend=dict(
            title=None,
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.48,
            font=dict(size=16)
        ),
        xaxis=dict(
            title=None,
            range=[-20,35],
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=None,
            range=[-20,30],
            tickfont=dict(size=14)
        )
    )

    st.plotly_chart(fig)

    with st.expander("About the Data"):
        st.markdown("""
        This UMAP plot reveals two prominent clusters — one predominantly SAVR and the other TAVR, suggesting that patients selected for each treatment type generally differ in their clinical profiles.

        However, there are also **regions where SAVR and TAVR patients mixed together**, highlighting a zone of clinical ambiguity. In these mixed regions, treatment decisions appear less driven by clear-cut features and more reliant on clinician judgment, preferences, or patient-specific considerations.
        """)


st.divider()




# ===========================
# Patient Profile and Insights
# ===========================

st.subheader("Patient Profile and Personalized Insights")

# generate a note
if not submit_btn:
    st.info("Please fill out the form in the sidebar or use a random patient profile to view personalized insights.")

# once button is clicked and all fields are filled
if submit_btn and all(user_input.values()):

    # display progress bar
    my_bar = st.progress(0, text="Loading models...")

    # load pre-trained model
    mu0_model = joblib.load('models/mu0_model.pkl')
    mu1_model = joblib.load('models/mu1_model.pkl')
    ps_model = joblib.load('models/ps_model.pkl')
    tau0_model = joblib.load('models/tau0_model.pkl')
    tau1_model = joblib.load('models/tau1_model.pkl')
    umap_model = joblib.load('models/umap_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    time.sleep(0.5)



    # process user input
    my_bar.progress(50, text="Processing user input...")

    user_input_df = pd.DataFrame([user_input])
    user_input_df = user_input_df.replace({'Yes':1, 'No':0, 'I':1, 'II':2, 'III':3, 'IV':4, 'Male':0, 'Female':1})
    for i in range(1,5):
        user_input_df[f'nyha_{i}'] = (user_input_df['nyha'] == i).astype(int)
    user_input_df = user_input_df.drop(columns=['nyha'])

    user_input_df[['age', 'lvef', 'plateletct']] = user_input_df[['age', 'lvef', 'plateletct']].astype(float)
    user_input_df[['age', 'height', 'weight', 'lvef', 'preproccreat', 'hgb', 'plateletct', 'albumin', 'inr']] = scaler.transform(user_input_df[['age', 'height', 'weight', 'lvef', 'preproccreat', 'hgb', 'plateletct', 'albumin', 'inr']])


    print(user_input_df)
    # get new embedding
    new_embedding = umap_model.transform(user_input_df)


    time.sleep(0.5)

    

    my_bar.progress(75, text="Plotting...")
    time.sleep(0.5)


    my_bar.progress(100, text="Done!")
    time.sleep(0.5)
    my_bar.empty() 

    ## Plot for TAVR patients
    df_tavr_plot = df_plot[df_plot['treatment'] == 'TAVR'].copy()

    # Separate patients
    talive = df_tavr_plot[df_tavr_plot['mace'] == 0]
    tdead = df_tavr_plot[df_tavr_plot['mace'] == 1]

    fig_talive = go.Scattergl(
        x=talive["UMAP1"],
        y=talive["UMAP2"],
        mode="markers",
        name="Without Adverse Event",
        marker=dict(
            color="#78C2FF",
            size=6,
            opacity=0.6
        )
    )

    fig_tdead = go.Scattergl(
        x=tdead["UMAP1"],
        y=tdead["UMAP2"],
        mode="markers",
        name="With Adverse Event",
        marker=dict(
            color="#FF4242",
            size=7,
            symbol="x",
            opacity=0.8
        )
    )

    fig_tnew = go.Scattergl(
        x=[new_embedding[0][0]],
        y=[new_embedding[0][1]],
        mode="markers",
        name="New Patient",
        marker=dict(
            color="#ffb703",
            size=20,
            symbol="star"
        )
    )

    tfig = go.Figure(data=[fig_talive, fig_tdead, fig_tnew])

    # Customize layout
    tfig.update_layout(
        height=600,
        legend=dict(
            title=None,
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.48,
            font=dict(size=16)
        ),
        xaxis=dict(
            title=None,
            range=[-20,35],
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=None,
            range=[-20,30],
            tickfont=dict(size=14)
        )
    )




    ## Plot for SAVR patients
    df_savr_plot = df_plot[df_plot['treatment'] == 'SAVR'].copy()

    # Separate patients
    salive = df_savr_plot[df_savr_plot['mace'] == 0]
    sdead = df_savr_plot[df_savr_plot['mace'] == 1]

    fig_salive = go.Scattergl(
        x=salive["UMAP1"],
        y=salive["UMAP2"],
        mode="markers",
        name="Without Adverse Event",
        marker=dict(
            color="#2F75C1",
            size=6,
            opacity=0.6
        )
    )

    fig_sdead = go.Scattergl(
        x=sdead["UMAP1"],
        y=sdead["UMAP2"],
        mode="markers",
        name="With Adverse Event",
        marker=dict(
            color="#FF4242",
            size=7,
            symbol="x",
            opacity=0.8
        )
    )

    fig_snew = go.Scattergl(
        x=[new_embedding[0][0]],
        y=[new_embedding[0][1]],
        mode="markers",
        name="New Patient",
        marker=dict(
            color="#ffb703",
            size=20,
            symbol="star"
        )
    )

    sfig = go.Figure(data=[fig_salive, fig_sdead, fig_snew])

    sfig.update_layout(
        height=600,
        legend=dict(
            title=None,
            orientation="h", 
            yanchor="bottom", 
            y=-0.2, 
            xanchor="center", 
            x=0.48,
            font=dict(size=16)
        ),
        xaxis=dict(
            title=None,
            range=[-20,35],
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=None,
            range=[-20,30],
            tickfont=dict(size=14)
        )
    )

    # Display in Streamlit
    st.markdown("**UMAP Projection of TAVR Patients**")
    st.markdown("The UMAP plot below shows the distribution of TAVR patients in the dataset. Each point represents a patient, colored and marked by whether they experienced a major adverse cardiovascular event (MACE) after the treatment. The star indicates your profile's position in this space.")
    st.plotly_chart(tfig, use_container_width=True)
    st.markdown("**UMAP Projection of SAVR Patients**")
    st.markdown("The UMAP plot below shows the distribution of SAVR patients in the dataset. Each point represents a patient, colored and marked by whether they experienced a major adverse cardiovascular event (MACE) after the treatment. The star indicates your profile's position in this space.")
    st.plotly_chart(sfig, use_container_width=True)




# ===========================
# Risk Contribution Analysis
# ===========================

# once button is clicked and all fields are filled
if submit_btn and all(user_input.values()):

    st.markdown("<br>", unsafe_allow_html=True)  # one line 
    st.markdown("**Baseline Patient Characteristic Contributions to TAVR / SAVR Risk**")

    explainer_mu0 = shap.Explainer(mu0_model)
    explainer_mu1 = shap.Explainer(mu1_model)

    # Get SHAP values for the new patient
    shap_values_mu0 = explainer_mu0(user_input_df)
    shap_values_mu1 = explainer_mu1(user_input_df)



    # Feature data
    features = ['Age', 'Gender', 'Height', 'Weight', 'LVEF', 'Heart Failure', 'Diabetes', 'Hypertension', 'Dialysis', 'COPD', 'CVD', 'Shock', 'MI', 'PAD', 'Creatinine', 'Hemoglobin', 'Platelet', 'Albumin', 'INR', 'NYHA Class I', 'NYHA Class II', 'NYHA Class III', 'NYHA Class IV']
    tavr_risk = shap_values_mu0.values[0].tolist()
    savr_risk = shap_values_mu1.values[0].tolist()

    tavr_risk = [round(x, 2) for x in tavr_risk]
    savr_risk = [round(x, 2) for x in savr_risk]
    

    # get range
    min = min(math.floor(min(tavr_risk)), math.floor(min(savr_risk)))
    min = -1.0 if min >= 0 else min
    max = max(math.ceil(max(tavr_risk)), math.ceil(max(savr_risk)))

    range = list(np.arange(min, max+1, 0.5))
    range_text = [str(round(x, 1)) for x in range]




    # Close loop
    tavr_risk += [tavr_risk[0]]
    savr_risk += [savr_risk[0]]

    angles = np.linspace(0, 360, len(tavr_risk))  # use degrees

    # Trace: 0 circle with high-resolution theta
    zero_line = go.Scatterpolar(
        r=[0] * 360,
        theta=np.linspace(0, 360, 360),
        mode='lines',
        line=dict(color="#FF4242", width=4, dash="dash"),
        hoverinfo='skip',
        showlegend=False
    )

    # Trace: actual radar data
    risk_trace = go.Scatterpolar(
        r=tavr_risk,
        theta=angles,
        mode='lines+markers',
        fill='toself',
        name="Risk Contribution to TAVR",
        marker=dict(size=6, color="#255D99"),
        line=dict(color='#255D99', width=2),
    )

    # Trace: actual radar data
    risk_trace2 = go.Scatterpolar(
        r=savr_risk,
        theta=angles,
        mode='lines+markers',
        fill='toself',
        name="Risk Contribution to SAVR",
        marker=dict(size=6, color="#80C4FB"),
        line=dict(color="#80C4FB", width=2),
    )

    # Create figure
    fig = go.Figure(data=[zero_line, risk_trace, risk_trace2])

    # Add feature names to angular axis
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=np.linspace(0, 360, len(features), endpoint=False),
                ticktext=features,
                direction="clockwise",
                rotation=90
            ),
            radialaxis=dict(
                visible=True,
                range=[min,max+0.5],
                tickvals=range,
                ticktext=range_text,
                gridcolor="#D3D3D3"
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        height=600,
        title=""
    )

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)


    # explanation text
    with st.expander("Explanation"):
        st.markdown("""
        This chart shows how each baseline characteristic affects your risk of experiencing a major adverse cardiovascular event (MACE) after TAVR or SAVR treatment.
        
        - **Positive value**: increases risk
        - **Negative value**: decreases risk
        - **Magnitude**: strength of contribution
        - **Zero**: no impact
        """)
        
    st.markdown("<br>", unsafe_allow_html=True)  # one line

    with st.container(border=True):

        # calculate risk scores
        mu0_new = mu0_model.predict_proba(user_input_df)[:, 1]
        tavr_prob = str(round(mu0_new[0]*100,2)) + "%"
        mu1_new = mu1_model.predict_proba(user_input_df)[:, 1]
        savr_prob = str(round(mu1_new[0]*100,2)) + "%"

        # Predict treatment effects
        tau0_new = tau0_model.predict(user_input_df)
        tau1_new = tau1_model.predict(user_input_df)

        # Predict propensity score
        propensity_scores = ps_model.predict_proba(user_input_df)[:, 1]

        # Combine τ0 and τ1 using weighting
        tau_new = propensity_scores * tau0_new + (1 - propensity_scores) * tau1_new
        if tau_new[0] < 0 :
            result_text = f"SAVR is expected to reduce your adjusted risk compared to TAVR by"
            result_value = str(round(abs(tau_new[0]*100),2)) + "%"
        else:
            result_text = f"SAVR is expected to increase your adjusted risk compared to TAVR by"
            result_value = str(round(abs(tau_new[0]*100),2)) + "%"



        # display results
        col1, col2, col3 = st.columns([1,1,2])
        col1.metric(label="TAVR Risk", value=tavr_prob)
        col2.metric(label="SAVR Risk", value=savr_prob)
        col3.metric(label=result_text, value=result_value)




        















   



















