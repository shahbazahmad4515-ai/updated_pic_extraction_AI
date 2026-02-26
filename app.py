import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
from huggingface_hub import hf_hub_download
from model_architecture import EncoderCNN, DecoderWithAttention, Vocabulary

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neural Storyteller v2", layout="wide", page_icon="🤖")

# Custom CSS for the AI look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .caption-box { padding: 20px; background-color: #238636; border-radius: 10px; color: white; font-size: 24px; font-weight: bold; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- HUGGING FACE DOWNLOAD ---
HF_REPO = "shahbazahmadshahbazahmad/image-caption-attention-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_assets():
    try:
        # This downloads the files from Hugging Face and saves them in a local cache
        vocab_path = hf_hub_download(repo_id=HF_REPO, filename="vocab.pkl")
        encoder_path = hf_hub_download(repo_id=HF_REPO, filename="encoder_att_v2_epoch_40.pth")
        decoder_path = hf_hub_download(repo_id=HF_REPO, filename="decoder_att_v2_epoch_40.pth")

        # 1. Load Vocab
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # 2. Load Encoder
        encoder = EncoderCNN().to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()
        
        # 3. Load Decoder
        decoder = DecoderWithAttention(
            attention_dim=256, 
            embed_size=256, 
            decoder_dim=256, 
            vocab_size=len(vocab)
        ).to(device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()
        
        return vocab, encoder, decoder
    except Exception as e:
        st.error(f"Critical Error loading model from Hugging Face: {e}")
        st.stop()

# Initialize the brain
vocab, encoder, decoder = load_assets()

# --- PREDICTION LOGIC ---
def generate_caption(image, encoder, decoder, vocab):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_out = encoder(img_tensor) # (1, 7, 7, 2048)
        encoder_out = encoder_out.view(1, -1, 2048)
        
        mean_encoder_out = encoder_out.mean(dim=1)
        h = decoder.init_h(mean_encoder_out)
        c = decoder.init_c(mean_encoder_out)
        
        start_token = vocab.stoi["<SOS>"]
        current_input = torch.tensor([start_token]).to(device)
        result_caption = []
        
        for t in range(25): # Increased to 25 words max
            embeddings = decoder.embedding(current_input)
            context_vector, alpha = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context_vector = gate * context_vector
            
            h, c = decoder.decode_step(torch.cat([embeddings, context_vector], dim=1), (h, c))
            preds = decoder.fc(h)
            predicted = preds.argmax(1)
            
            word = vocab.itos[predicted.item()]
            if word == "<EOS>": break
            if word != "<SOS>": result_caption.append(word)
            current_input = predicted
            
    return ' '.join(result_caption)

# --- UI LAYOUT ---
st.title("🧠 Neural Storyteller: Advanced Attention Model")
st.markdown("This AI uses **Spatial Attention** to focus on specific parts of an image for higher precision.")

# Sidebar Metrics
with st.sidebar:
    st.header("📈 Model Statistics")
    st.write("Results after 40 Epochs:")
    st.metric("BLEU-4 Score", "0.2140") 
    st.metric("Vocabulary Size", f"{len(vocab)}")
    st.metric("Architecture", "ResNet50 + Attention")
    st.write("---")
    st.markdown("**Status:** Deployment Successful ✅")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

with col2:
    st.subheader("✨ AI Interpretation")
    if uploaded_file:
        if st.button('Describe This Scene'):
            with st.spinner('Analyzing visual regions...'):
                caption = generate_caption(image, encoder, decoder, vocab)
                st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)
                st.write("---")
                st.info("💡 **Attention Logic:** The model identifies regions of interest to describe textures, colors, and subjects with advanced spatial awareness.")
    else:
        st.warning("Please upload an image to see the AI in action.")

st.markdown("---")
st.caption("University Project | Developed by Shahbaz | Powered by PyTorch, Hugging Face & Streamlit")
