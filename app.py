import time
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from methods import Methods
from load import Load

def analyze_single_matrix(matrix):
    matrix_latex = Load.latex_conversion(matrix)
    st.latex(f"\Large A = {matrix_latex}")
    power, inverse = st.columns(2)
    with power:
        st.scatter_chart(results['power-method-values'][0], x_label='Iterations', y_label='Eigenvalue', height=500)
        st.latex(r"\Large \lambda_0 \approx" + str(results['power-method-values'][0][-1]))
    with inverse:
        st.scatter_chart(results['inverse-power-method-values'][0], x_label='Iterations', y_label='Eigenvalue',height=500)
        st.latex(r"\Large \lambda_1 \approx" + str(results['inverse-power-method-values'][0][-1]))


st.set_page_config(layout="wide")

homepage = st.empty()

example_dataset = pd.read_csv('assets/dataset.csv')

with homepage:

    text, img = st.columns((3, 1))

    with text:
        text.write("""
        # Welcome to Eigen-Analyzer
        Eigen-Analyzer is designed to help you analyze and compute dominant eigenvalues using the **Power Method** and its variant the **Inverse Power Method**.
        
        ## How to Use
        1. **Upload a CSV file**: The file should contain a column named `matrix` with the matrices to be analyzed.
        2. **Set Parameters**: Use the sidebar to set the tolerance and the number of iterations.
        3. **Analyze**: Once the file is uploaded, the analysis will start automatically.
        """)

    with img:
        animation = Load.load_lottiefile("assets/animation.json")
        st_lottie(animation, height=300, width=300)

st.sidebar.title("Parameters")
tolerance = st.sidebar.number_input("Tdolerance", value=1e-6, format="%.6f", step=1e-6)
iterations = int(st.sidebar.slider("Number of Iterations",min_value=1,max_value=100,value=10,step=1))
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

results = None

if file is not None:
    df = pd.read_csv(file)
    if 'matrix' not in df.columns:
        st.error("Please upload a valid CSV file.")
    else:
        valid = True
        for index, row in df.iterrows():
            try:
                matrix = np.array(eval(row['matrix']))
            except:
                valid = False
                break
        if not valid:
            st.error("The matrix column contains invalid data.")
        else:
            with st.sidebar, st.spinner("Analyzing data..."):
                time.sleep(3)
                results = Methods.analyze_data(df, tolerance, iterations)


if results is not None:

    if len(results) == 0:
        st.error("No valid matrices were found in the dataset.")
        st.stop()

    homepage.empty()

    if len(results) == 1:
        matrix = np.array(eval(results['matrix'][0]))
        analyze_single_matrix(matrix)


    else :
        dominant = [v[len(v) - 1] for v in results['power-method-values']]
        results['power-method-iterations'] = [len(v) for v in results['power-method-values']]
        results['inverse-power-method-iterations'] = [len(v) for v in results['inverse-power-method-values']]

        st.write("""
            # Summary Statistics
            ##### The summary statistics provide an overview of the data distribution and convergence behavior.
        """)

        col1, col2 = st.columns([0.75,2.75])

        with col1:
            st.write(results.describe())
        with col2:
            st.write(results.head(8))

        st.write("""
            ## Dominant Eigenvalues
        """)
        st.scatter_chart(dominant, x_label='Matrix-Index', height=500)

        st.write("""
            # Number of Iterations Required to Converge to the Dominant Eigenvalue
            ##### The Inverse Power Method is a modification of the Power Method that gives faster convergence. It is not common to see the number of iterations required to converge for the Power Method to be less than the Inverse Power Method.
        """)

        power, inverse = st.columns(2)

        with power:
            st.write("""
                ### Power Method
            """)
            st.bar_chart(results['power-method-iterations'], x_label='Matrix-Index', y_label='Iterations', height=500)

        with inverse:
            st.write("""
                ### Inverse Power Method
            """)
            st.bar_chart(results['inverse-power-method-iterations'], x_label='Matrix-Index', y_label='Iterations', height=500)

        st.write("""
            # Error Reduction on Estimation
            ##### To show the error reduction on both methods, we will take a matrix who has a number of iterations for the power-method equal to the mean.
        """)

        row = results.iloc[int(results['power-method-iterations'].mean())]
        matrix = np.array(eval(row['matrix']))
        analyze_single_matrix(matrix)