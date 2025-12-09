# 📄 Denoising Dirty Documents using U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

> **Final Project - Deep Learning**
> A Deep Learning approach to restore and clean noisy scanned documents for better OCR readability.

## Group 8 - LB01
* Jessica Priscilla Immanuel (2702246163)
* Marsha Genevieve Nandana (2702217522)
* Ricardo Cuthbert (2702353612)

## 🖼️ Project Overview

Digitizing old or handwritten documents often results in "dirty" images filled with stains, paper wrinkles, shadows, and noise. This degradation makes it difficult for Optical Character Recognition (OCR) systems to read the text accurately.

This project implements a **U-Net Autoencoder** to perform Image Denoising. The model learns to map "dirty" input images to their "clean" ground truth versions, effectively removing background noise while preserving text sharpness.

## ‼ Key Features

* **Custom U-Net Architecture:** An encoder-decoder network optimized for image restoration.
* **Patch-Based Training Strategy:** Instead of resizing full documents (which destroys text resolution), we train on random **256x256 patches**. This ensures the model learns high-frequency details.
* **Seamless Stitching Inference:** The Streamlit app implements a "sliding window" prediction technique to process full A4 pages without distortion or resizing artifacts.
* **High Performance:** Achieved an **SSIM of ~0.9981** and **PSNR of ~38 dB**.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, NumPy
* **Web App:** Streamlit
* **OCR Testing:** PyTesseract

