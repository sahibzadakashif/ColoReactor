# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model
model = joblib.load("model.pkl")

# Streamlit page settings
st.set_page_config(page_title="ParkiC50", layout="wide")

# Function to generate Morgan fingerprints
def smiles_to_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    return None

# Function to predict pIC50 and classify
def predict_pIC50_and_class(smiles_list):
    results = []
    for smi in smiles_list:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            pIC50 = model.predict([fp])[0]
            activity = (
                "Active" if pIC50 >= 6 else
                "Intermediate" if pIC50 >= 5 else
                "Inactive"
            )
            results.append((smi, round(pIC50, 2), activity))
        else:
            results.append((smi, None, "Invalid SMILES"))
    return pd.DataFrame(results, columns=["SMILES", "Predicted pIC50", "Bioactivity Class"])

# Function to create CSV download link
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'

# Main Streamlit app
def main():
    # Load and display image
    st.image("2427039e-791d-409b-9026-481140c81f5e.png", use_column_width=True)

    # Title and subtitle
    st.markdown("<h1 style='text-align: center; color: #8E44AD;'>🧠 ParkiC50</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size:18px;'>"
        "AI-powered prediction of <b>pIC₅₀</b> values and bioactivity classification for "
        "<b>Protein Kinase C</b> targeting drug candidates in humans against Parkinson's Disease."
        "</p>", 
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("🔬 Sequence Submission")

    input_method = st.radio("Choose Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_input = st.text_area("Please Enter SMILES Strings (one per line):")
        if st.button("Predict"):
            smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
            if not smiles_list:
                st.warning("⚠️ Please enter valid SMILES.")
            else:
                df = predict_pIC50_and_class(smiles_list)
                st.success("✅ Prediction complete!")
                st.dataframe(df, use_container_width=True)
                st.markdown(get_download_link(df), unsafe_allow_html=True)
    else:
        file = st.file_uploader("Upload a CSV or TXT file containing SMILES strings", type=["csv", "txt"])
        if file and st.button("Predict"):
            try:
                df = pd.read_csv(file, header=None)
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
                results = predict_pIC50_and_class(smiles_list)
                st.success("✅ Prediction complete!")
                st.dataframe(results, use_container_width=True)
                st.markdown(get_download_link(results), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.subheader("👨‍🔬 ParkiC50 Development Team")

    # Developer profiles
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div style='line-height: 1.6; text-align: center;'>
                <h4 style='color:#8E44AD;'>Dr. Kashif Iqbal Sahibzada</h4>
                Assistant Professor<br>
                Department of Health Professional Technologies,<br>
                The University of Lahore<br>
                Post-Doctoral Fellow, Henan University of Technology, China<br>
                <b>Email:</b><br>kashif.iqbal@dhpt.uol.edu.pk<br>kashif.iqbal@haut.edu.cn
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='line-height: 1.6; text-align: center;'>
                <h4 style='color:#8E44AD;'>Shumaila Shahid</h4>
                MS Biochemistry<br>
                School of Biochemistry and Biotechnology,<br>
                University of the Punjab, Lahore<br>
                <b>Email:</b><br>shumaila.ms.sbb@pu.edu.pk
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='line-height: 1.6; text-align: center;'>
                <h4 style='color:#8E44AD;'>Fajar Abbas</h4>
                School of Biochemistry and Biotechnology,<br>
                University of the Punjab, Lahore<br>
                <b>Email:</b><br>fajarabbas433@gmail.com
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div style='line-height: 1.6; text-align: center;'>
                <h4 style='color:#8E44AD;'>Saher Abbas</h4>
                School of Biochemistry and Biotechnology,<br>
                University of the Punjab, Lahore<br>
                <b>Email:</b><br>saherabbas272004@gmail.com
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
