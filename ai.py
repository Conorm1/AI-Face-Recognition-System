import os
import cv2
import random

# Load the pre-trained face recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the path to the folder containing images
folder_path = 'AiProject/archive/lfw-deepfunneled/lfw-deepfunneled/Abdullah'

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Print the contents of the image_files list
print("Image files:", image_files)

# Check if there are any files in the folder
if not image_files:
    print("No image files found in the specified folder.")
else:
    # Randomly select an image
    selected_image = random.choice(image_files)

    # Load the selected image
    image_path = os.path.join(folder_path, selected_image)
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        # Loop through detected faces and do further processing
        for (x, y, w, h) in faces:
            # Do something with each face, like draw a rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow('AI Facial Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error loading image: {image_path}")


