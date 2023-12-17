# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st
from tensorflow.keras.preprocessing import image
# Local Modules
import settings
import helper
import numpy as np
import cv2

models_name = {'Grape': ['Grape__black_measles', 'Grape__black_rot', 'Grape__healthy'],
                'Guava': ['guava_Disease Free', 'guava_Phytopthora', 'guava_Red rust', 'guava_Scab', 'guava_Styler and Root'],
                'Lemon': ['Lemon__diseased', 'Lemon__healthy'],
                'Mango': ['mango_Anthracnose', 'mango_Bacterial Canker','mango_Cutting Weevil', 'mango_Die Back', 
                        'mango_Gall Midge','mango_Healthy','mango_Powdery Mildew','mango_Sooty Mould'],
                'Pomegranate': ['Pomegranate__diseased', 'Pomegranate__healthy']}
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


# Create a list of options for the dropdown
Fruits_options = list(models_name.keys())


# Display the selected option
st.sidebar.header('Fruits and Flowers')

# Add a dropdown with text before the selection
selected_option = st.sidebar.selectbox('Select an Fruit:', Fruits_options)


if model_type == 'Classify':
    model_path = Path(settings.MODEL_DIR / selected_option / 'best_model_weights.h5')
    print(model_path)

cls = models_name[selected_option]

print(selected_option, len(cls), cls)

try:
    # st.caching.clear_cache()
    st.cache_data.clear()
    model = helper.load_model(model_path,selected_option, len(cls))
    st.success("Model Loaded successfully!")
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
            st.header("The Disease is: " + 'Bacterial Canker', )

        else:
            if st.sidebar.button('Detect Objects'):

                img = uploaded_image.resize((256, 256)) 
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0 

                res = model.predict(img_array)
                print(res[0])
                predicted_class_index = np.argmax(res)
                st.header("The Disease is: " + cls[predicted_class_index])
                st.header("The Confidance is: " + str(res[0][predicted_class_index]))
                

else:
    st.error("Please select a valid source type!")
