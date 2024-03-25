import cv2
import numpy as np


class Cropper:

    def __init__(self):
        # Initialize CUDA-accelerated OpenCV
        cv2.cuda.setDevice(0)  # Set the GPU device (change if you have multiple GPUs)

    def crop_image(self, image_path, bounding_boxes):
        # Load image
        gpu_image = self.load_image(image_path)

        # Crop bounding boxes
        cropped_images = self.crop_bounding_boxes(gpu_image, bounding_boxes)

        return cropped_images

    @staticmethod
    def load_image(image_path):
        # Read the image
        image = cv2.imread(image_path)

        # Convert image to GPU mat
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        return gpu_image

    @staticmethod
    def crop_bounding_boxes(gpu_image, bounding_boxes):
        stream = cv2.cuda_Stream()
        cropped_images = []
        for box in bounding_boxes:
            # Extract bounding box coordinates
            x, y, w, h = box

            # Create a GPU mat for the bounding box
            gpu_box = cv2.cuda_GpuMat(gpu_image, y, x, h, w)

            # Download the cropped region from GPU to CPU
            cropped_gpu_box = cv2.cuda_GpuMat()
            cropped_gpu_box.upload(gpu_box, stream=stream)

            # Convert GPU mat to numpy array
            cropped_image = cropped_gpu_box.download(stream=stream)
            cropped_images.append(cropped_image)

        # Synchronize CUDA stream
        stream.waitForCompletion()

        return cropped_images
