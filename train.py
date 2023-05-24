import csv
import os

from sklearn.model_selection import train_test_split

# emotion = input('Enter the emotion: ')
emotions = ['Anger', 'Disgust', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
for emotion in emotions:
    input_directory = f'dataset_resized/{emotion}'
    # Get a list of all image files in the input directory
    image_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory)
                   if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')]

    # Create a list of corresponding labels (assuming the emotion folder name is the label)
    labels = [emotion] * len(image_files)

    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)

    # Print the number of images in each set
    print(f"Number of training images for {emotion}:", len(train_images))
    print(f"Number of testing images for {emotion}:", len(test_images))



    # Define the output file paths
    train_output_file = 'train_data.csv'
    test_output_file = 'test_data.csv'

    # Save training data to a CSV file
    with open(train_output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(train_images, train_labels))

    # Save testing data to a CSV file
    with open(test_output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(test_images, test_labels))