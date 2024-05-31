import pickle

def save_processor(processor):
    with open('processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    print("Processor saved to disk.")

def load_processor():
    with open('processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    return processor

# Example of how to use it after training
if __name__ == "__main__":
    # Load the processor (assuming it's already loaded or defined after fine-tuning)
    processor = load_processor()

    # Call the function to save the processor
    save_processor(processor)
