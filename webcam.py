import cv2
import torch
import torchvision.transforms as transforms
from emotion_detection_model import EmotionDetectionModel

num_classes = 7
# Load the trained model
model = EmotionDetectionModel(num_classes)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Set up the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
target_size = 128  # Set the target size for resizing the frames

# Preprocess function for webcam frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
])

# Run the webcam loop
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        break

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(frame)
        predicted_class = torch.argmax(predictions, dim=1).item()

    # Display the frame and predicted label
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
