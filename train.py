# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# --- Configuration ---
# Define the paths and training parameters

# Path to the dataset configuration file (data.yaml)
# Make sure this path is correct relative to where you run the script
DATA_YAML_PATH = 'datasets/data.yaml'

# Pre-trained model to start from (e.g., 'yolov8n.pt' for nano version)
# The library will automatically download it if not found locally
BASE_MODEL = 'yolov8n.pt'

# Training parameters
NUM_EPOCHS = 150     # Number of training epochs
IMAGE_SIZE = 640     # Input image size (square)
BATCH_SIZE = 32      # Number of images per batch

# --- Main Training Function ---
def train_yolo_model():
    """
    Loads the YOLO model and starts the training process
    using the specified configuration.
    """
    print("--- Starting YOLOv8 Training ---")
    print(f"Dataset YAML: {DATA_YAML_PATH}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-" * 30)

    try:
        # Load a pre-trained YOLO model
        # For training, you typically start from a pre-trained model like yolov8n.pt
        model = YOLO(BASE_MODEL)

        # Train the model using the specified dataset and parameters
        # The results (including trained weights, logs, etc.) will be saved
        # in a 'runs/detect/train*' directory by default.
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=NUM_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            # You can add other arguments here, e.g.:
            patience=0,  # Early stopping patience
            device=0,     # Specify GPU device (e.g., 0) or 'cpu'
            # name='my_custom_training_run' # Name for the output directory
        )

        print("-" * 30)
        print("--- Training Finished ---")
        print(f"Results saved to: {results.save_dir}") # Access the save directory if needed

    except FileNotFoundError:
        print(f"Error: Dataset configuration file not found at '{DATA_YAML_PATH}'.")
        print("Please ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

# --- Script Execution ---
if __name__ == '__main__':
    # This block ensures the training function runs only when
    # the script is executed directly (not imported as a module).
    train_yolo_model()
