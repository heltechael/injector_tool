import torch
import torch.nn as nn
import torch.optim as optim
import time

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image

class ThumbnailAssessor:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = models.resnet18()  # Create an instance of the model architecture
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model = model.to(self.device)
        return model

    def preprocess_thumbnail(self, thumbnail_image):
        # Convert the NumPy array to a PIL image
        thumbnail_image = Image.fromarray(thumbnail_image)

        # Resize the thumbnail image to 224x224 while maintaining aspect ratio
        resized_image = transforms.functional.resize(thumbnail_image, (224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Apply padding to fill the remaining space
        padded_image = transforms.functional.pad(resized_image, padding=0, fill=0)
        
        # Apply the same preprocessing transformations used during training
        processed_image = self.transform(padded_image)
        return processed_image

    def assess(self, thumbnail_image):
        # Measure preprocessing time
        preprocess_start_time = time.time()
        processed_image = self.preprocess_thumbnail(thumbnail_image)
        preprocess_end_time = time.time()
        preprocess_time = preprocess_end_time - preprocess_start_time

        # Measure model inference time
        inference_start_time = time.time()
        processed_image = processed_image.unsqueeze(0)  # Add batch dimension
        processed_image = processed_image.to(self.device)
        with torch.no_grad():
            prediction = self.model(processed_image)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        assessment_score = torch.softmax(prediction, dim=1)[0][0].item()  # Assuming the "good" class is at index 0

        # Print the time taken by each step
        print(f"Preprocessing time: {preprocess_time:.4f} seconds")
        print(f"Model inference time: {inference_time:.4f} seconds")

        return assessment_score