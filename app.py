import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import math

#load model yang udh dilatih
@st.cache_resource
def load_modell():
    model = load_model("denoising_unet_model_lighter.h5")
    return model

model = load_modell()

def process(model, image, patch_size=256):
    img = ImageOps.grayscale(image)
    img_array = np.array(img)
    
    h, w = img_array.shape
    
    pad_h = (math.ceil(h / patch_size) * patch_size) - h
    pad_w = (math.ceil(w / patch_size) * patch_size) - w
    
    img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='edge')
    new_h, new_w = img_padded.shape
    
    reconstructed_img = np.zeros_like(img_padded, dtype=float) 
    
    patches = []
    coords = []
    
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = img_padded[i:i+patch_size, j:j+patch_size]
            patch_norm = patch.astype('float32') / 255.0
            patch_input = np.expand_dims(patch_norm, axis=-1)
            patches.append(patch_input)
            coords.append((i, j))
            
    patches_array = np.array(patches) 
    predictions = model.predict(patches_array, verbose=0)
    
    for idx, (i, j) in enumerate(coords):
        pred_patch = predictions[idx]
        pred_patch = np.squeeze(pred_patch)
        reconstructed_img[i:i+patch_size, j:j+patch_size] = pred_patch
        
    
    final_img = reconstructed_img[:h, :w] #crop balik ke ukuran asli
    
    #ganti (0.0 - 1.0) jadi pixel (0 - 255), jadi ga maksa > 0.5 jadi item/putih
    final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(final_img)




#streamlit
st.set_page_config(
    page_title="Denoising Dirty Documents",
    page_icon="📃",
    layout="wide"
)

#title
st.markdown(
    f"""
    <h1 style="text-align:center; margin-bottom:0;">📃 Denoising Dirty Documents 🔍</h1>

    <p style="text-align:center; font-size:1.05rem; color:gray; margin-top:6px;">
       Upload dokumen kotor untuk dibersihkan menggunakan model U-Net.
    </p>

    <p style="text-align:center; font-size:0.95rem; color:#666; ">
        Model ini dapat membuat dokumen bersih dari bayangan, noda, dll sehingga dapat memaksimalkan sistem Optical Character Recognition (OCR).
       
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

#upload gambrr
uploaded = st.file_uploader("Upload gambar dokumen yang mau dilakukan denoising", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    col1, col2 = st.columns(2)

    #tampilin yang asli, yg bru di upload
    with col1:
        st.header("Dokumen Asli")
        st.write("Dokumen yang masih memiliki banyak noise dan belum dibersihkan.")

        #tampilin gambar
        img = Image.open(uploaded)
        st.image(img, caption="Original Document", width=420)

        tanda = 0

        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #BCC5E0; 
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
        if st.button("Clean"):
            with st.spinner("Tunggu..."): #process denoising
                clean_img = process(model, img)
            
            tanda = 1
            st.success("Selesai!")

    #tampilin yang udah bersih
    with col2:
        st.header("Dokumen Bersih")
        st.write("Dokumen yang sudah tidak ada noise dan mudah dibaca.")
        
        if tanda == 1:
            st.image(clean_img, caption="Denoised Result", width=420)

            clean_img.save("cleaned.png")
            
            #download gambar yg udah bersih
            with open("cleaned.png", "rb") as file: 
                st.markdown("""
                    <style>
                        .stDownloadButton>button {
                            background-color: #BBDAED; 
                            color: white;
                        }
                    </style>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="Download Dokumen",
                    data=file,
                    file_name="cleaned_document.png",
                    mime="image/png"
                )