import streamlit as st
from io import StringIO
import sys
import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def highlight_code(code):
    return pygments.highlight(code, PythonLexer(), HtmlFormatter())

def main():
    st.write("### Питон компилятор* на странице в стримлит")

    # Default ASCII art code
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
    codecopy.replace('\n','\n\n')
    
    st.write("##### Что видит стримлит ")
    
    highlighted_code = highlight_code(codecopy)
    #st.markdown(highlighted_code, unsafe_allow_html=True)
    st.code(highlighted_code)
    
    st.write("##### Результат выполнения ")

    if st.button("Запустить"):
        try:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            
            exec(st.session_state.code)
            
            sys.stdout = old_stdout
            
            result = redirected_output.getvalue()
            st.code(result)
        
        except Exception as e:
            st.error(f"Err: {str(e)}")
    
    st.write("#### Как сделать так же ")
    
    displaymenow = """
            
            ###### Использованные библиотеки
            
                import streamlit as st
                from io import StringIO
                import sys
                import pygments
                from pygments.lexers import PythonLexer
                from pygments.formatters import HtmlFormatter
            
            ###### Подсветка кода
            
                def highlight_code(code):
                    return pygments.highlight(code, PythonLexer(), HtmlFormatter())
                    
                ...
                highlighted_code = highlight_code(codecopy)
            
            ###### Код в окне по умолчанию

                default_code = \"\"\" я код по умолчанию  \"\"\"
                
                ...
                code = st.text_area(\"Введите код на питоне:\", height=300, value=default_code)

            ###### Обращение к окну с именем code
            
                if st.button(\"Отчистить окно\"):
                    st.session_state.code = ""
            
            ###### Что видит стримлит
 
                codecopy = code
                codecopy.replace(\'\\n\' , \'\\n\\n\')
                
                highlighted_code = highlight_code(codecopy)
                st.code(highlighted_code)

            ###### Запуск
            
                if st.button(\"Запустить\"):
                    try:
                        old_stdout = sys.stdout
                        redirected_output = sys.stdout = StringIO()
                        
                        exec(st.session_state.code)
                        
                        sys.stdout = old_stdout
                        
                        result = redirected_output.getvalue()
                        st.code(result)
                    
                    except Exception as e:
                        st.error(f\"Err: {str(e)}\")

            
            """
    #displaymenow = highlight_code( displaymenow)
    
    st.write(displaymenow)



main()
