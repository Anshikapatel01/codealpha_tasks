import streamlit as st
import requests

st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌍"
)

st.title("🌍 English to Hindi Translator")
st.write("Translate English text into Hindi for free.")

text = st.text_area(
    "Enter English text",
    placeholder="How are you today?",
    height=150
)

if st.button("Translate to Hindi 🌐", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Translating..."):
                response = requests.get(
                    "https://api.mymemory.translated.net/get",
                    params={
                        "q": text,
                        "langpair": "en|hi"
                    },
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()

            if data.get("responseStatus") == 200:
                translation = data["responseData"]["translatedText"]
                st.subheader("Hindi Translation")
                st.text_area(
                    "Translated text",
                    value=translation,
                    height=150
                )
            else:
                st.error("Translation service limit reached or unavailable.")

        except requests.RequestException:
            st.error("Please check your internet connection and try again.")