# Test of DeepFace
# Created by Jiaxiao Shi on 2024/05/15

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# List of emotions and their corresponding filenames
emotions = ['happy', 'sad', 'sad1', 'angry', 'disgust', 'surprise', 'fear', 'fear1', 'fear2', 'neutral']

# Iterate over each emotion
for emotion in emotions:
    # Load an image
    img_path = f"./test_img/{emotion}.jpg"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Analyze the image
    results = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion', 'race'])
    print(results)
    
    # Initialize lists to store results for each face
    face_info = []

    # Process results for each face
    for result in results:
        face_info.append({
            "age": result["age"],
            "dominant_gender": max(result["gender"], key=result["gender"].get),
            "dominant_emotion": max(result["emotion"], key=result["emotion"].get),
            "dominant_race": max(result["race"], key=result["race"].get)
        })

    # Display the image
    plt.imshow(img_rgb)
    plt.text(100, 250, f'Analyse Img - {emotion.capitalize()}', fontsize=14, color='red', weight='bold')
    plt.axis('off')

    # Add text annotations for each face at the bottom of the image
    for i, info in enumerate(face_info):
        text = f'Age - {info["age"]}\nGender - {info["dominant_gender"]}\nEmotion - {info["dominant_emotion"]}\nRace - {info["dominant_race"]}'
        plt.text(100, img_rgb.shape[0] - 120 - 100*i, text, fontsize=13, color='black', weight='bold')

    # Save the annotated image with the corresponding emotion name
    result_filename = f"./result_img/result_{emotion}.jpg"
    plt.savefig(result_filename, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to release resources

    print(f"Result saved as {result_filename}\n")

print("\nAll images processed and saved successfully.\n")