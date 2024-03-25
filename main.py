import argparse

from cropper import Cropper


def parse_args():
    parser = argparse.ArgumentParser(description='Cropper')

    # Define command-line arguments
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('device', type=str, default=0, args="?", help='device')

    return parser.parse_args()


if __file__ == "__main__":
    args = parse_args()

    bounding_boxes = [(0, 0, 100, 100), (100, 100, 200, 200), (200, 200, 300, 300), (300, 300, 400, 400)]
    cropper = Cropper(device_name=args.device)
    cropped_images = cropper.crop_image(args.image_path, bounding_boxes)
    print(cropped_images)
