import streamlit as st
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Set page configuration
st.set_page_config(page_title="Image Captioning App", page_icon="🖼️", layout="wide")

# Initialize session state for queries and images
if "queries" not in st.session_state:
    st.session_state["queries"] = []

if "images" not in st.session_state:
    st.session_state["images"] = []

# Title and file uploader
st.title("🖼️ Image Captioning App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Upload your image here")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption for the uploaded image
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            image_buffer = io.BytesIO(uploaded_file.getvalue())
            # Convert the image buffer to PIL Image
            image_pil = Image.open(image_buffer)
            # Process the image and generate the caption
            inputs = processor(images=image_pil, return_tensors="pt")
            outputs = model.generate(**inputs)
            response = processor.decode(outputs[0], skip_special_tokens=True)
            st.success(f"**Response:** {response}")  # Only display response

            # Store the query and response in session state
            st.session_state["queries"].append({"question": "What is the image description?", "response": response})
            st.session_state["images"].append(uploaded_file.name)

    # Sidebar to show history of queries
    st.sidebar.title("📝 Query History")
    if st.session_state["queries"]:
        for idx, query in enumerate(st.session_state["queries"]):
            st.sidebar.write(f"**Image:** {st.session_state['images'][idx]}")
            st.sidebar.write(f"**Query:** {query['question']}")
            st.sidebar.write(f"**Response:** {query['response']}")
