import time
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from methods import Methods
from load import Load

def analyze_single_matrix(matrix, results):
    matrix_latex = Load.latex_conversion(matrix)
    st.latex(f"\Large A = {matrix_latex}")


    st.write("""
        #### Convergence Procedure
    """)
    y1 = np.array(results['power-method-values'][0])
    y2 = np.array(results['inverse-power-method-values'][0])
    x = np.arange(start=0, stop=max(y1.size, y2.size), step=1)

    while y1.size < x.size:
        y1 = np.append(y1, y1[-1])
    while y2.size < x.size:
        y2 = np.append(y2, y2[-1])

    data = pd.DataFrame({
        'x': x,
        'power-method': y1,
        'inverse-method': y2
    })

    st.line_chart(data.set_index('x'))

    pow_, inv_ = st.columns(2)

    with pow_:
        st.latex(r"\lambda_0 \approx" + str(results['power-method-values'][0][-1]))

    with inv_:
        st.latex(r"\lambda_1 \approx" + str(results['inverse-power-method-values'][0][-1]))

    if len(results['power-vector']) > 10:
        st.error("""
            ##### The dimension of the eigen-vector is too large to be displayed
        """)
    else:
        st.latex(Load.latex_vector(results['power-vector'], 'v_0'))
        st.latex(Load.latex_vector(results['inverse-vector'], 'v_1'))


def main():
    st.set_page_config(layout="wide")

    homepage = st.empty()

    with homepage:

        text, img = st.columns((3, 1))

        with text:
            text.write("""
            # Welcome to Eigen-Analyzer
            Eigen-Analyzer is a powerful yet intuitive tool designed to explore the fascinating world of matrices. Created with educational research in mind, this app helps you dive deeper into the mathematical concepts of eigenvalues and eigenvectors.
            At its core, Eigen-Analyzer harnesses the Power Method and its variant, the Inverse Power Method, two fundamental algorithms taught in Numerical Analysis. These methods drive the calculations and provide insights into matrix behavior, making this app an excellent companion for students mastering these topics.
            
            Whether you’re a student, educator, or math enthusiast, Eigen-Analyzer is here to enhance your understanding of linear algebra and numerical analysis, making your exploration both insightful and engaging.
            
            ##### Discover the elegance of mathematics with Eigen-Analyzer!
            
            ## How to Use
            1. **Upload a CSV file**: The file should contain a column named `matrix` with the matrices to be analyzed.
            2. **Set Parameters**: Use the sidebar to set the tolerance and the number of iterations.
            3. **Analyze**: Once the file is uploaded, the analysis will start automatically.
            """)

        with img:
            animation = Load.load_lottiefile("assets/animation.json")
            st_lottie(animation, height=350, width=300)

    st.sidebar.title("Parameters")
    tolerance = st.sidebar.number_input("Tolerance", value=1e-6, format="%.6f", step=1e-6)
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
            analyze_single_matrix(matrix, results)


        else :
            dominant = [v[len(v) - 1] for v in results['power-method-values']]
            results['power-method-iterations'] = [len(v) for v in results['power-method-values']]
            results['inverse-power-method-iterations'] = [len(v) for v in results['inverse-power-method-values']]

            st.write("""
                ## Summary Statistics
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
                ## Number of Iterations Required to Converge to the Dominant Eigenvalue
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

            row = results.iloc[int(results['power-method-iterations'].mean())]
            matrix = np.array(eval(row['matrix']))
            analyze_single_matrix(matrix, results)

if __name__ == '__main__':
    main()