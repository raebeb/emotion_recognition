import os
from skimage import io, img_as_ubyte
from skimage.transform import resize


def resize_images(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                image = io.imread(image_path)
                normalized_image = normalize_pixels(image)  # Normalize pixel values
                resized_image = resize(normalized_image, (target_size, target_size))
                resized_image = img_as_ubyte(resized_image)  # Convert back to uint8 format
                io.imsave(output_path, resized_image)
                print(f"Resized {filename} successfully.")
            except Exception as e:
                print(f"Failed to resize {filename}: {str(e)}")


def normalize_pixels(image):
    """
    Normalize the pixel values of the image to the range [0, 1].
    """
    normalized_image = image.astype(float) / 255.0
    return normalized_image


emotion = input('Enter the emotion to resize (Anger, Disgust, Happiness, Neutral, Sadness): ')
# Example usage
input_directory = f'dataset/{emotion}'
output_directory = f'dataset_resized/{emotion}'
target_size = 224

resize_images(input_directory, output_directory, target_size)
