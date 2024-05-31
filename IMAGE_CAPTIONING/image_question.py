import pickle
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_buffer, text=None):
    try:
        # Load the processor from disk
        with open('processor.pkl', 'rb') as f:
            processor = pickle.load(f)

        # Load the model from disk
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        image = Image.open(image_buffer).convert('RGB')
        if text:
            inputs = processor(image, text, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"
