# import streamlit as st
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import sys
# import os

# # ---------- Fix Python Path ----------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ---------- Import Models ----------
# from models import SimpleAutoencoder, ImprovedAutoencoder, UNet, DnCNN

# # ---------- Import Patch Restoration ----------
# from utils.patch_utils import restore_large_image

# from utils.corruption_utils import add_gaussian_noise
   
# device = torch.device("cpu")

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# # ---------- Transform ----------
# to_tensor = transforms.ToTensor()

# # ---------- Load Models ----------
# @st.cache_resource
# def load_models():

#     simple = SimpleAutoencoder()
#     simple.load_state_dict(
#         torch.load(os.path.join(MODELS_DIR, "simple_autoencoder_model.pth"), map_location=device)
#     )
#     simple.eval()

#     improved = ImprovedAutoencoder()
#     improved.load_state_dict(
#         torch.load(os.path.join(MODELS_DIR, "improved_autoencoder_model.pth"), map_location=device)
#     )
#     improved.eval()

#     unet = UNet()
#     unet.load_state_dict(
#         torch.load(os.path.join(MODELS_DIR, "unet_model.pth"), map_location=device)
#     )
#     unet.eval()

#     dncnn = DnCNN()
#     dncnn.load_state_dict(
#         torch.load(os.path.join(MODELS_DIR, "dncnn_model.pth"), map_location=device)
#     )
#     dncnn.eval()

#     return simple, improved, unet, dncnn


# simple_model, improved_model, unet_model, dncnn_model = load_models()

# # ---------- Streamlit UI ----------

# st.title("Clear Vision - Image Restoration")

# st.write("Upload an image and choose a model to restore it.")

# model_choice = st.selectbox(
#     "Choose Model",
#     ["Simple Autoencoder", "Improved Autoencoder", "U-Net", "DnCNN"]
# )


# uploaded_file = st.file_uploader(
#     "Upload Image",
#     type=["png", "jpg", "jpeg"]
# )

# # ---------- Image Processing ----------

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     st.subheader("Original Image")
#     st.image(image)

#     img_tensor = to_tensor(image)

#     corrupted = add_gaussian_noise(img_tensor)

   

#     # select model
#     if model_choice == "Simple Autoencoder":
#         model = simple_model

#     elif model_choice == "Improved Autoencoder":
#         model = improved_model

#     elif model_choice == "U-Net":
#         model = unet_model

#     else:
#         model = dncnn_model

#     # ---------- Patch Based Restoration ----------
#     restored_tensor = restore_large_image(model, corrupted)



#     st.subheader("Corrupted Image")
#     st.image(corrupted.permute(1,2,0).numpy())

#     st.subheader("Restored Image")
#     st.image(restored_tensor.permute(1,2,0).numpy())

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# ---------- Fix Python Path ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- Import Models ----------
from models import SimpleAutoencoder, ImprovedAutoencoder, UNet, DnCNN

# ---------- Import Patch Restoration ----------
from utils.patch_utils import restore_large_image

# ---------- Import Gaussian Noise ----------
from utils.corruption_utils import add_gaussian_noise

device = torch.device("cpu")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------- Transform ----------
to_tensor = transforms.ToTensor()

# ---------- PSNR Function ----------
def calculate_psnr(original, restored):

    original = original.numpy()
    restored = restored.numpy()

    mse = np.mean((original - restored) ** 2)

    if mse == 0:
        return 100

    psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    return psnr


# ---------- Load Models ----------
@st.cache_resource
def load_models():

    simple = SimpleAutoencoder()
    simple.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "simple_autoencoder_model.pth"), map_location=device)
    )
    simple.eval()

    improved = ImprovedAutoencoder()
    improved.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "improved_autoencoder_model.pth"), map_location=device)
    )
    improved.eval()

    unet = UNet()
    unet.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "unet_model.pth"), map_location=device)
    )
    unet.eval()

    dncnn = DnCNN()
    dncnn.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "dncnn_model.pth"), map_location=device)
    )
    dncnn.eval()

    return simple, improved, unet, dncnn


simple_model, improved_model, unet_model, dncnn_model = load_models()

# ---------- Streamlit UI ----------

st.title("Clear Vision - Image Restoration")

st.markdown("""
Upload an image to evaluate how different deep learning models remove **Gaussian noise**.

        Models compared:
            
            • Basic Denoising Autoencoder 
            • Deep Denoising Autoencoder 
            • U-Net Restoration Network
            • Residual Denoising CNN
""")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

# ---------- Image Processing ----------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")


    img_tensor = to_tensor(image)

    # ---------- Add Gaussian Noise ----------
    corrupted = add_gaussian_noise(img_tensor)



    # ---------- Restore with All Models ----------
    simple_restored = restore_large_image(simple_model, corrupted)
    improved_restored = restore_large_image(improved_model, corrupted)
    unet_restored = restore_large_image(unet_model, corrupted)
    dncnn_restored = restore_large_image(dncnn_model, corrupted)

    simple_restored = torch.clamp(simple_restored,0,1)
    improved_restored = torch.clamp(improved_restored,0,1)
    unet_restored = torch.clamp(unet_restored,0,1)
    dncnn_restored = torch.clamp(dncnn_restored,0,1)

    # ---------- Calculate PSNR ----------
    simple_psnr = calculate_psnr(img_tensor, simple_restored)
    improved_psnr = calculate_psnr(img_tensor, improved_restored)
    unet_psnr = calculate_psnr(img_tensor, unet_restored)
    dncnn_psnr = calculate_psnr(img_tensor, dncnn_restored)

    # ---------- Display Results ----------
    st.subheader("Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(corrupted.permute(1,2,0).numpy(), caption="Corrupted Image")

    # ---------- Display Model Comparison ----------
    st.subheader("Model Comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(simple_restored.permute(1,2,0).numpy())
        st.markdown(f"**Simple Autoencoder**  \nPSNR: {simple_psnr:.2f} dB")

    with col2:
        st.image(improved_restored.permute(1,2,0).numpy())
        st.markdown(f"**Deep Autoencoder**  \nPSNR: {improved_psnr:.2f} dB")

    with col3:
        st.image(unet_restored.permute(1,2,0).numpy())
        st.markdown(f"**U-Net**  \nPSNR: {unet_psnr:.2f} dB")

    with col4:
        st.image(dncnn_restored.permute(1,2,0).numpy())
        st.markdown(f"**DnCNN**  \nPSNR: {dncnn_psnr:.2f} dB")