import json

import requests

class Load:
    @staticmethod
    def latex_conversion(matrix):
        latex_matrix = r'\begin{pmatrix}' + \
                       r'\\'.join([' & '.join(map(str, row)) for row in matrix]) + \
                       r'\end{pmatrix}'
        return latex_matrix


    @staticmethod
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()