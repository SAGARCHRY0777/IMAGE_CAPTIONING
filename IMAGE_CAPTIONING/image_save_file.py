import pickle
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Save processor and model to disk
with open('processor.pkl', 'wb') as f:
    pickle.dump(processor, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Processor and model saved to disk.")
