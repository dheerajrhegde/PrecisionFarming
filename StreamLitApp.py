import geocoder
import streamlit as st
from keras.preprocessing import image

import PrecisionFarming


def update_city(lat, long):
    """Update the city and state based on latitude and longitude."""
    arc = geocoder.arcgis([lat, long], method="reverse")
    st.session_state.loc = f"{arc.city}, {arc.state}"


def update_lat_long(location):
    """Update the latitude and longitude based on the location."""
    st.session_state.lat, st.session_state.long = geocoder.arcgis(location).latlng


# Set up the page configuration
st.set_page_config(
    page_title="Precision Farming",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Precision Farming")

if "pf" not in st.session_state:
    pf = PrecisionFarming.PrecisionFarming()
    st.session_state["pf"] = pf
else:
    pf = st.session_state["pf"]

# Setting up layout
col1, col2 = st.columns([0.2, 0.8])

if "loc" not in st.session_state:
    st.session_state.loc = "Concord, NC"
    st.session_state.lat, st.session_state.long = geocoder.arcgis(st.session_state.loc).latlng

with col1:
    loc = st.text_input("Location", key="loc", on_change=lambda: update_lat_long(st.session_state.loc))
    lat = st.number_input("Latitude", step=0.0001, key="lat",
                          on_change=lambda: update_city(st.session_state.lat, st.session_state.long))
    long = st.number_input("Longitude", step=0.0001, key="long",
                           on_change=lambda: update_city(st.session_state.lat, st.session_state.long))

    with st.form(key="farmer_data", clear_on_submit=False):
        ph = st.number_input("Soil pH", value=6.5, step=0.1)
        moisture = st.number_input("Soil Moisture", value=30, step=1)
        area = st.number_input("Area (acres)", value=10, step=1)
        crop = st.selectbox("What crop do you want to get information for?", ("Corn", "Soybean", "Cotton"))

        latitude = st.session_state.lat
        longitude = st.session_state.long

        insect = st.file_uploader("Upload an image of an insect", type=["jpg", "png"])
        leaf = st.file_uploader("Upload an image of a leaf", type=["jpg", "png"])

        submitted = st.form_submit_button("Get Insights")

with col2:
    if submitted and insect and leaf:
        insect_img = image.load_img(insect, target_size=(224, 224))
        leaf_img = image.load_img(leaf, target_size=(224, 224))
        insights = pf.get_insights(ph, moisture, latitude, longitude, area, crop, insect_img, leaf_img)
        st.markdown(insights)
    else:
        st.markdown("Please fill out the form to get insights.")
