import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(image_path, model_path='weapons.pt', confidence_threshold=0.5):
    """
    Detect objects in an image using a YOLO model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO model weights (default: yolov8n.pt)
        confidence_threshold (float): Minimum confidence score for detections (0-1)
    
    Returns:
        tuple: Processed image with boxes, list of detections
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Run inference
    desired_classes = [1]  
    results = model(image, classes=desired_classes)[0]
    
    # Process detections
    detections = []
    
    # Get the original image for drawing
    output_image = image.copy()
    
    # Process each detection
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        
        if confidence >= confidence_threshold:
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_name = results.names[int(class_id)]
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add detection to list
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    return output_image, detections

def main():
    # Example usage
    image_path = 'test_image_of_gun.jpg'
    
    try:
        # Process image
        processed_image, detections = detect_objects(image_path)
        
        # Display results
        print("\nDetections:")
        for i, detection in enumerate(detections, 1):
            print(f"\nDetection {i}:")
            print(f"Class: {detection['class']}")
            print(f"Confidence: {detection['confidence']:.2f}")
            print(f"Bounding Box: {detection['bbox']}")
        
        # Show image
        cv2.imshow('Object Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save the processed image
        cv2.imwrite('output.jpg', processed_image)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()