import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class CNN_Retino(nn.Module):
    def __init__(self):
        super(CNN_Retino, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Calculate the output size of the convolutional layers
        self._to_linear = None
        self._get_conv_output(torch.randn(1, 3, 128, 128))  # Assuming 128x128 images

        # Fully connected layers (to be manually initialized if there's a mismatch)
        self.fc1 = nn.Linear(self._to_linear, 100)  # Adjust this layer size if needed
        self.fc2 = nn.Linear(100, 2)  # Final layer for binary classification

    def _get_conv_output(self, dummy_input):
        """ This function calculates the size of the feature map after the conv layers. """
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # Max pooling after conv layers
        x = self.dropout1(x)  # Apply dropout
        x = torch.flatten(x, 1)  # Flatten the output for fully connected layers
        self._to_linear = x.size(1)  # Store the flattened size for use in the first FC layer

    def forward(self, x):
        """ The forward pass through the network """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = self.dropout1(x)  # Dropout
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected layer
        x = self.dropout2(x)  # Dropout
        x = self.fc2(x)  # Final output
        return x


# Load the model and weights
model = CNN_Retino()

model_path = r'C:\Users\notan\OneDrive\Desktop\DR\Retino_model_weights.pt'

if not os.path.exists(model_path):
    print(f"Error: The model file {model_path} does not exist.")
else:
    try:
        # Load only the convolutional layers' weights
        state_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # Filter out the convolutional weights from the loaded state_dict
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(filtered_dict)  # Update the model's state_dict

        model.load_state_dict(model_dict)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


# Image transformation function for preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to 128x128
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet-style normalization
])

# Class names for prediction
class_names = ['No_DR', 'DR']

def predict_image(image_path):
    absolute_path = os.path.abspath(image_path)
    print(f"Checking file at: {absolute_path}")

    if not os.path.exists(image_path):
        print(f"Error: The file {absolute_path} does not exist.")
        return

    # Open and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        
        # Debugging: print raw logits
        print(f"Raw logits for {os.path.basename(image_path)}: {output}")
        
        probabilities = F.softmax(output, dim=1)
        
        # Debugging: print probabilities after softmax
        print(f"Probabilities: {probabilities}")
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

    # Print the prediction results
    print(f"\nPredicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    return (os.path.basename(image_path), class_names[predicted_class], confidence)

# Save predictions to CSV
def save_predictions_to_csv(predictions, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Predicted Class", "Confidence"])
        for prediction in predictions:
            writer.writerow(prediction)

# Test the prediction
if __name__ == "__main__":
    sample_folder = r"C:\Users\notan\OneDrive\Desktop\DR\sample"
    output_csv_path = r"C:\Users\notan\OneDrive\Desktop\DR\predictions.csv"

    if not os.path.exists(sample_folder):
        print(f"Error: The folder {sample_folder} does not exist.")
    else:
        # Loop through all image files in the folder
        supported_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(sample_folder)
                       if os.path.splitext(f)[1].lower() in supported_extensions]

        if not image_files:
            print("No images found in the folder.")
        else:
            predictions = []
            for image_file in image_files:
                image_path = os.path.join(sample_folder, image_file)
                print(f"\n--- Predicting: {image_file} ---")
                prediction = predict_image(image_path)
                if prediction:
                    predictions.append(prediction)

            # Save predictions to CSV file
            save_predictions_to_csv(predictions, output_csv_path)
            print(f"\nPredictions saved to: {output_csv_path}")
