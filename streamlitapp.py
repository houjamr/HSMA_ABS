import streamlit as st

# Title of the app
st.title("Streamlit App Example")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(["Home", "Page 1", "Page 2"])

# Home Page
if page == "Home":
    st.header("Welcome to the Home Page!")
    st.write("This is the main landing page of the app.")
    st.image("https://via.placeholder.com/300", caption="Placeholder Image")

# Page 1
elif page == "Page 1":
    st.header("Page 1: Columns Example")
    col1, col2 = st.columns(2)
    col1.write("This is content in Column 1.")
    col2.write("This is content in Column 2.")

# Page 2
elif page == "Page 2":
    st.header("Page 2: Tabs Example")
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("Welcome to Tab 1!")
        st.text_input("Enter some text for Tab 1:")
    with tab2:
        st.write("Welcome to Tab 2!")
        st.slider("Select a value for Tab 2:", 0, 100, 50)

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit")

