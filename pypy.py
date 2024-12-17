import streamlit as st
from io import StringIO
import sys
import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

import numpy as np
from scipy.optimize import root
import scipy

def highlight_code(code):
    return pygments.highlight(code, PythonLexer(), HtmlFormatter())

def execute_python_code(code):
    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        exec(code)
        
        sys.stdout = old_stdout
        result = redirected_output.getvalue()
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        sys.stdout = old_stdout

def main():
    st.write("### Питон компилятор* на странице в стримлит")

    default_code = """
    # ASCII https://www.asciiart.eu/animals/rabbits
def draw_me():
    cat = '''
                    /|      __
    *             +      / |   ,-~ /             +
        .              Y :|  //  /                .         *
            .          | jj /( .^     *
                *    >-"~"-v"              .        *        .
    *                  /       Y
    .     .        jo  o    |     .            +
                    ( ~T~     j                     +     .
        +           >._-' _./         +
                /| ;-"~ _  l
    .           / l/ ,-"~    \     +
                \//\/      .- \
        +       Y        /    Y
                l       I     !
                ]\      _\    /"\  
                (" ~----( ~   Y.  )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    print(cat)

draw_me()
    """

    code = st.text_area("Введите код на питоне:", height=300, value=default_code)
    
    if st.button("Отчистить окно"):
        st.session_state.code = ""
    
    st.session_state.code = code
    codecopy = code
    codecopy.replace('\n', '\n\n')
    
    st.write("##### Результат выполнения ")

    if st.button("Запустить"):
        result = execute_python_code(st.session_state.code)
        st.code(result)
        
    st.write("##### Что видит стримлит ")
    
    highlighted_code = highlight_code(codecopy)
    st.code(highlighted_code)
