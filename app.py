import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import math
import pytesseract  # Tambahan untuk OCR

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
uploaded = st.file_uploader("Upload gambar dokumen", type=["jpg", "jpeg", "png"])

#session state biar hasil gak ilang pas klik tombol lain
if 'clean_img' not in st.session_state:
    st.session_state.clean_img = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = ""

if uploaded is not None:
    col1, col2 = st.columns(2)

    #tampilin yang asli, yg bru di upload
    with col1:
        st.header("Dokumen Asli")
        st.write("Dokumen awal yang memiliki noise.")
        img = Image.open(uploaded) #tampilin gambar
        st.image(img, caption="Dokumen Asli", width=420)

        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #BCC5E0; 
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("Bersihkan"):
            with st.spinner("Sedang membersihkan & membaca teks..."):
                cleaned = process(model, img) #process denoising
                st.session_state.clean_img = cleaned
                
                #OCR 
                # config='--psm 6' diasumsikan blok teks seragam, bisa dihapus kl error
                text_result = pytesseract.image_to_string(cleaned) 
                st.session_state.ocr_text = text_result
            
            st.success("Selesai!")

    #yg udh bersih
    with col2:
        st.header("Hasil Pemrosesan")
        
        #session state
        if st.session_state.clean_img is not None:
            tab_img, tab_ocr = st.tabs(["🖼️ Dokumen Bersih", "📝 Hasil OCR (Teks)"])

            st.markdown("""
                <style>
                    .stDownloadButton>button {
                        background-color: #88DAED; 
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            with tab_img:
                st.write("Dokumen yang sudah bersih dari noise.")
                st.image(st.session_state.clean_img, caption="Hasil Denoised U-Net", width=420)
                
                # Tombol Download Gambar
                from io import BytesIO
                buf = BytesIO()
                st.session_state.clean_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Gambar",
                    data=byte_im,
                    file_name="cleaned_document.png",
                    mime="image/png"
                )

            with tab_ocr:
                st.write("Teks yang berhasil dibaca dari gambar bersih:")
                #biar bs dicpy
                st.text_area("Teks dalam Dokumen", st.session_state.ocr_text, width=420)
                
                #download text
                st.download_button(
                    label="Download Teks (.txt)",
                    data=st.session_state.ocr_text,
                    file_name="document_text.txt",
                    mime="text/plain"
                )
