import streamlit as st
import numpy as np
from scipy.optimize import root_scalar

import streamlit as st
from front_page import show_front_page

import task4_1
import task4_2
import task8_1
import task8_2
import task12_1
import task12_2


def countdown_timer():
    return """
    <div id="countdown-timer" style="position: fixed; top: 100px; right: 100px; background-color: black; border: 1px solid black; padding: 5px; font-size: 16px;">
        <span id="timer">01:00:00</span>
    </div>

    <script>
        let countDownDate = new Date().getTime() + 3600000; // Add 1 hour
        let x;

        function updateCountdown() {
            let now = new Date().getTime();
            let distance = countDownDate - now;

            if (distance <= 0) {
                clearInterval(x);
                document.getElementById("timer").innerHTML = "EXPIRED";
                return;
            }

            let hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
            let minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
            let seconds = Math.floor((distance % (1000 * 60)) / 1000);

            document.getElementById("timer").innerHTML = String(hours).padStart(2, "0") + ":" +
                String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
        }

        updateCountdown();
        x = setInterval(updateCountdown, 1000);
    </script>
    """

if 'task' not in st.session_state:
    st.session_state.task = "Главная страница"

with st.sidebar:
    # Front Page Link
    st.title("Навигация")
    #if st.button("Front Page"):
    #    st.session_state.task = "Front Page"
        
    # Task Selection
    st.title("Выбор страницы")
    selected_task = st.selectbox("Выберете страницу", ["Главная страница", "Задача 4_1", "Задача 4_2", "Задача 8_1", "Задача 8_2", "Задача 12_1", "Задача 12_2"])
    if selected_task:
        st.session_state.task = selected_task

# Main content area
if st.session_state.task == "Главная страница":
    show_front_page()
elif st.session_state.task == "Задача 4_1":
    task4_1.run_task4_1()
elif st.session_state.task == "Задача 4_2":
    task4_2.run_task4_2()
elif st.session_state.task == "Задача 8_1":
    task8_1.run_task8_1()
elif st.session_state.task == "Задача 8_2":
    task8_2.run_task8_2()
elif st.session_state.task == "Задача 12_1":
    task12_1.run_task12_1()
elif st.session_state.task == "Задача 12_2":
    task12_2.run_task12_2()
else:
    st.error("Invalid task selection")

# Add countdown timer
st.markdown(countdown_timer(), unsafe_allow_html=True)

# Adjust Streamlit's layout to accommodate the scroll-to-top button
st.markdown("""
<style>
.stApp {
    padding-left: 40px !important;
}
</style>
""", unsafe_allow_html=True)

# Add custom CSS for the scroll-to-top button
st.markdown("""
<style>
.scroll-to-top {
    position: fixed;
    bottom: 20px;
    right: 40px;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# Add JavaScript for scrolling to top
st.markdown("""
<script>
function scrollToTop() {
    window.scrollTo({top: 0, behavior: 'smooth'});
}
</script>
""", unsafe_allow_html=True)

# Add the scroll-to-top button
html_button = """
<button onclick="scrollToTop()" class="scroll-to-top" style="background-color:#0066cc;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-up-circle-fill" viewBox="0 0 16 16">
        <path d="M16 8A8 8 0 1 0 0 8a8 8 0 0 0 16 0zm-7.5 3.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V11.5z"/>
    </svg> Scroll to Top
</button>
"""
st.markdown(html_button, unsafe_allow_html=True)