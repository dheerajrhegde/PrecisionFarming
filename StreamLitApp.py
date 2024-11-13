import streamlit as st
import time
import geocoder
from keras.preprocessing import image
from PrecisionFarming import PrecisionFarming
from streamlit_folium import st_folium
from streamlit_modal import Modal
import requests
import folium


def show_completion_animation():
    """Show a farming-themed emoji animation."""
    success_emojis = ["🌱", "🌿", "🌾", "🎋", "✨"]
    animation_placeholder = st.empty()
    for emoji in success_emojis:
        animation_placeholder.markdown(
            f"<h1 style='text-align: center'>{emoji}</h1>", 
            unsafe_allow_html=True
        )
        time.sleep(0.3)
    animation_placeholder.empty()

def update_city(lat, long):
    """Update the city and state based on latitude and longitude."""
    arc = geocoder.arcgis([lat, long], method="reverse")
    st.session_state.loc = f"{arc.city}, {arc.state}"

def update_lat_long(location):
    """Update the latitude and longitude based on the location."""
    st.session_state.lat, st.session_state.long = geocoder.arcgis(
        location).latlng

def create_streamlit_app():
    st.set_page_config(page_title="Precision Farming Agent",
                       page_icon="🌾",
                       layout="wide")
    
    st.title("🌾 Precision Farming Agent")

    modal = Modal("Locate Your Field", key="map_modal")


    # State variables for selected location and location name
    if "selected_location" not in st.session_state:
        st.session_state["selected_location"] = ""
    if "location_name" not in st.session_state:
        st.session_state["location_name"] = ""


    if "pf" not in st.session_state:
        st.session_state.pf = PrecisionFarming()
    if "loc" not in st.session_state:
        st.session_state.loc = "Concord, NC"
        st.session_state.lat, st.session_state.long = geocoder.arcgis(
            st.session_state.loc).latlng

    st.header("Field Parameters")
    input_col, status_col = st.columns(2)

    with input_col:
        
        loc = st.text_input("Location", 
                           key="loc",
                           value=st.session_state["selected_location"],
                           on_change=lambda: update_lat_long(st.session_state.loc))
        
        # Button to open the modal
        if st.button("Locate Your Field"):
            modal.open()

            # Show the modal and load the map inside it when the button is clicked
        if modal.is_open():
            with modal.container():
                lattitude, longitude = geocoder.arcgis(
                st.session_state.loc).latlng
                # Set default location and zoom level for the map
                map_center = [lattitude, longitude]  # Centered on India (for example)
                zoom_start = 12

                # Create the map with folium
                m = folium.Map(location=map_center, zoom_start=zoom_start)

                # Add an event to capture clicks on the map and return latitude and longitude
                m.add_child(folium.LatLngPopup())

                # Display the map in Streamlit using st_folium
                map_data = st_folium(m, width=900, height=600)

                # Display the selected latitude and longitude
                if map_data["last_clicked"]:
                    lat = map_data["last_clicked"]["lat"]
                    lon = map_data["last_clicked"]["lng"]
                    arc = geocoder.arcgis([lat, lon], method="reverse")
                    locationName = f"{arc.city}, {arc.state}"
                    st.session_state["selected_location"] = locationName
                    # lat1, long1 = geocoder.arcgis(
                    # st.session_state.selected_location).latlng
                    st.session_state["lat"] = lat
                    st.session_state["long"] = lon
                    print(f"Selected Location: {locationName} &&  Latitude = {lat}, Longitude = {lon}")
                    modal.close()


        lat_col, long_col = st.columns(2)
        with lat_col:
            latitude = st.number_input("Latitude",
                                     value=st.session_state.lat,
                                     on_change=lambda: update_city(
                                         st.session_state.lat,
                                         st.session_state.long))
        with long_col:
            longitude = st.number_input("Longitude",
                                      value=st.session_state.long,
                                      on_change=lambda: update_city(
                                          st.session_state.lat,
                                          st.session_state.long))

        area = st.number_input("Area (acres)", min_value=1, value=10)
       
        leaf_img = st.file_uploader("Upload Leaf Disease Image (optional)",
                                   type=['jpg', 'jpeg', 'png'])
        if leaf_img:
            st.image(leaf_img, caption="Uploaded Leaf Image")
            leaf_img = image.load_img(leaf_img, target_size=(224, 224))

    with status_col:
        crop = st.selectbox("Crop", ["Corn", "Cotton", "Soybean"])
        soil_ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
        soil_moisture = st.slider("Soil Moisture %", 0, 100, 30, 1)
        
        insect_img = st.file_uploader("Upload Insect Image (optional)",
                                     type=['jpg', 'jpeg', 'png'])
        if insect_img:
            st.image(insect_img, caption="Uploaded Insect Image")
            insect_img = image.load_img(insect_img, target_size=(224, 224))

 
    analyze_button = st.button("Analyze Farm Conditions", on_click=lambda: display_area.empty())
    
    display_area = st.empty()

    if analyze_button:
            # Create a container within the placeholder for progress
            with display_area.container():
                status_area = st.container()
                progress_area = st.container()
                tool_progress = {}

                def update_status(message, progress, tool_name):
                    with status_area:
                        if progress == -1:
                            st.error(message)
                            return

                        if tool_name not in tool_progress:
                            with progress_area:
                                tool_progress[tool_name] = {
                                    'status': st.empty(),
                                    'bar': st.progress(0)
                                }

                        prog = tool_progress[tool_name]
                        prog['status'].info(f"{tool_name}: {message}")
                        prog['bar'].progress(int(progress))

                try:

                    # Get insights with progress tracking
                    insights = st.session_state.pf.get_insights(
                        soil_ph=soil_ph,
                        soil_moisture=soil_moisture,
                        latitude=latitude,
                        longitude=longitude,
                        area_acres=area,
                        crop=crop,
                        insect=insect_img,
                        leaf=leaf_img,
                        status_callback=update_status)

                    # Show completion animation
                    show_completion_animation()

                    # Clear the progress display and show results
                    display_area.empty()
                    with display_area.container():
                        st.header("Analysis Status & Results")
                        st.success("✨ Analysis Complete!")
                        st.markdown("### Farm Analysis Results")
                        # Add a visual separator
                        st.markdown("---")
                        # Display the results in a clean format
                        st.markdown(insights)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()