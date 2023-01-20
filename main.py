import streamlit as st
import os
import shutil
from tqdm import tqdm
from glob import glob
from zipfile import ZipFile
from io import BytesIO
import base64

from function import *

INPUT_FOLDER_PATH = 'saved_images'
OUTPUT_FOLDER_PATH = 'predicted_masks'
MODEL_PARAMETERS_PATH = 'best_model.pth' # soybean masks
MODEL_PARAMETERS_2_PATH = 'best_model_2.pth' # soybean and stage masks

if os.path.isdir(INPUT_FOLDER_PATH):
    shutil.rmtree(INPUT_FOLDER_PATH) # Delete a saved images folder
os.makedirs(INPUT_FOLDER_PATH, exist_ok=True) # Make a folder for saved images
    
if os.path.isdir(OUTPUT_FOLDER_PATH):
    shutil.rmtree(OUTPUT_FOLDER_PATH) # Delete a predicted masks folder
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True) # Make a folder for predicted masks

st.title('Mask Prediction')

uploaded_files = st.file_uploader('Please upload your image files', type=['png', 'jpg'], accept_multiple_files=True)
latest_iteration = st.empty()
if uploaded_files is not None:
    for i in range(len(uploaded_files)):
        latest_iteration.text(f'Uploading images {i+1}')

selected_type = st.radio(
    'Please select which mask type you predict',
    ['soybean', 'soybean & stage']
)
if selected_type == 'soybean':
    st.write('You selected "soybean".')
else:
    st.write('You selected "soybean & stage".')
            
start = st.button('Start prediction')    
 
if start:
    if os.path.isdir(INPUT_FOLDER_PATH):
        shutil.rmtree(INPUT_FOLDER_PATH) # Delete a saved images folder
    os.makedirs(INPUT_FOLDER_PATH, exist_ok=True) # Make a folder for saved images
    
    if os.path.isdir(OUTPUT_FOLDER_PATH):
        shutil.rmtree(OUTPUT_FOLDER_PATH) # Delete a predicted masks folder
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True) # Make a folder for predicted masks
    
    for uploaded_file in uploaded_files:
        with open(os.path.join(INPUT_FOLDER_PATH,uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer()) 
            
    if uploaded_files:  
        img_path_list = glob(os.path.join(INPUT_FOLDER_PATH, '*'))

        # Set variables for model
        DATA_DIR = img_path_list
        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'softmax2d'
        DEVICE = 'cuda' # cpu or cuda
    
        dataset = create_dataset(DATA_DIR, ENCODER, ENCODER_WEIGHTS)
        
        if selected_type == 'soybean':
            CLASSES = ['background', 'soybean', 'stake']
            model = prepare_model(ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, DEVICE, MODEL_PARAMETERS_PATH)
            inference(model, dataset, DATA_DIR, CLASSES, DEVICE, OUTPUT_FOLDER_PATH)
            ZIP_FILE_PATH = 'soybean_masks.zip'
        else:
            CLASSES = ['background', 'soybean', 'stake', 'stage']
            model = prepare_model(ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, DEVICE, MODEL_PARAMETERS_2_PATH)
            inference_2(model, dataset, DATA_DIR, CLASSES, DEVICE, OUTPUT_FOLDER_PATH)
            ZIP_FILE_PATH = 'soybean_stage_masks.zip'
        
        succeeded_process = st.success('Finished!')
        
        if succeeded_process:
            zipObj = ZipFile(ZIP_FILE_PATH, "w")
            predicted_masks = glob(os.path.join(OUTPUT_FOLDER_PATH, '*'))
            for predicted_mask in predicted_masks:
                zipObj.write(predicted_mask)
                os.remove(predicted_mask) # Delete predicted mask file
            zipObj.close()

            with open(ZIP_FILE_PATH, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZIP_FILE_PATH}'>\
                    Click last model weights\
                </a>"
            
            with open(ZIP_FILE_PATH, "rb") as fp:
                btn = st.download_button(
                    label="Download masks",
                    data=fp,
                    file_name=ZIP_FILE_PATH,
                    mime="application/zip"
                )
            
            os.remove(ZIP_FILE_PATH) # Delete a zip file in current directory
            shutil.rmtree(INPUT_FOLDER_PATH) # Delete a saved images folder
            shutil.rmtree(OUTPUT_FOLDER_PATH) # Delete a predicted masks folder
                        
    else:
        st.error('Please upload your image files')

uploaded_files = st.empty()