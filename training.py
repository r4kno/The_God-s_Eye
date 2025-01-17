from ultralytics import YOLO
import os
import torch

def train_yolo_model(
    data_yaml_path,
    model_path='yolov8m.pt',  # Path to your pre-downloaded model
    epochs=100,
    imgsz=640,
    batch_size=16,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # use '0' for first GPU, 'cpu' for CPU
):
    """
    Train a YOLOv8 model on a custom dataset using a pre-downloaded model file.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file from Roboflow
        model_path (str): Path to your pre-downloaded YOLOv8 model file
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch_size (int): Batch size for training
        device (str): Device to train on ('0' for GPU, 'cpu' for CPU)
    """
    # Initialize model from your pre-downloaded file
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=50,  # Early stopping patience
        save=True,    # Save best model
        plots=True    # Generate training plots
    )
    
    print(f"Training completed. Model saved in: {os.path.dirname(results.save_dir)}")
    return results

if __name__ == "__main__":
    # Example usage
    data_yaml_path = "C:/Users/ASUS/Downloads/Guns.v4i.yolov8/data.yaml"  # Replace with your data.yaml path
    model_path = "yolov8m.pt"     # Replace with your model path
    
    results = train_yolo_model(
        data_yaml_path=data_yaml_path,
        model_path=model_path,
        epochs=100,
        batch_size=16
    )