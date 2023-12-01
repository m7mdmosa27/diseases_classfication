# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
# Local Modules
import settings
import helper
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="ML Classification Models For Diseases",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("ML Classification Models For Diseases")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Classify'])

# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Classify':
    model_path = Path(settings.CLASSIFY_MODEL)
# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)

# Create a list of options for the dropdown
options = ['Guava', 'Mango']


# Display the selected option
st.sidebar.header('Fruits and Flowers')

# Add a dropdown with text before the selection
selected_option = st.sidebar.selectbox('Select an Fruit:', options)

if selected_option == 'Mango':
    cls = ['guava_Disease Free', 'guava_Phytopthora', 'guava_Red rust', 'guava_Scab', 'guava_Styler and Root']

elif selected_option == 'Guava':
    cls = ['guava_Disease Free', 'guava_Phytopthora', 'guava_Red rust', 'guava_Scab', 'guava_Styler and Root']

# elif selected_option == 'Mango':
#     cls = ['guava_Disease Free', 'guava_Phytopthora', 'guava_Red rust', 'guava_Scab', 'guava_Styler and Root']

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            # default_detected_image = PIL.Image.open(
            #     default_detected_image_path)
            # st.image(default_detected_image_path, caption='Detected Image',
            #          use_column_width=True)
            st.header("The Disease is: " + 'Bacterial Canker', )

        else:
            if st.sidebar.button('Detect Objects'):

                img = uploaded_image.resize((256, 256)) 
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0 

                res = model.predict(img_array)
                # boxes = res[0].boxes
                # res_plotted = res[0].plot()[:, :, ::-1]
                # st.image(res_plotted, caption='Detected Image',
                #          use_column_width=True)
                # try:
                #     with st.expander("Detection Results"):
                #         for box in boxes:
                #             st.write(box.data)
                # except Exception as ex:
                #     # st.write(ex)
                #     st.write("No image is uploaded yet!")
                predicted_class_index = np.argmax(res)
                st.header("The Disease is: " + cls[predicted_class_index])
                st.header("The Confidance is: " + str(res[0][predicted_class_index]))

# elif source_radio == settings.VIDEO:
#     helper.play_stored_video(confidence, model)

# elif source_radio == settings.WEBCAM:
#     helper.play_webcam(confidence, model)

# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
