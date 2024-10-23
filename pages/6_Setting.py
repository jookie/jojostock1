import streamlit as st

def main():
    st.title("Settings Page")

    # Create a form
    with st.form(key='settings_form'):
        st.header("User Information")
        
        # User information section
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        st.header("Preferences")
        # Preferences section
        theme = st.selectbox("Theme", ["Light", "Dark"])
        notifications = st.checkbox("Enable notifications", value=True)
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        
        st.header("Application Settings")
        # Application settings section
        autosave = st.slider("Autosave interval (minutes)", min_value=1, max_value=30, value=5)
        upload_limit = st.number_input("Max upload file size (MB)", min_value=1, max_value=1000, value=50)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Save Settings')

    # If the form is submitted, display the inputs
    if submit_button:
        st.success("Settings saved successfully!")
        st.write("### Summary of your settings:")
        st.write(f"**Username:** {username}")
        st.write(f"**Email:** {email}")
        st.write(f"**Theme:** {theme}")
        st.write(f"**Notifications Enabled:** {notifications}")
        st.write(f"**Language:** {language}")
        st.write(f"**Autosave Interval:** {autosave} minutes")
        st.write(f"**Max Upload File Size:** {upload_limit} MB")

if __name__ == "__main__":
    main()
