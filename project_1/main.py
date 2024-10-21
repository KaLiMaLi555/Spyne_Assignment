import os
from argparse import ArgumentParser

from utils import Data, load_image, replace_background, save_image


def setup_folders():
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)


if __name__ == "__main__":
    setup_folders()

    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Path to data folder")
    parser.add_argument("--image", type=str, help="Path to output folder")
    parser.add_argument("--all", action="store_true", help="Process all images")
    args = parser.parse_args()
    data_folder = args.data_folder

    if args.all:
        data_objs = []
        for image in os.listdir(os.path.join(data_folder, "images")):
            image_id = image.split(".")[0]
            if image_id.isnumeric():
                data_objs.append(Data(data_folder, image_id))
    elif args.image is not None:
        data_objs = [Data(data_folder, args.image)]
    else:
        raise ValueError("Provide either --all or --image")

    print(f"Processing {len(data_objs)} images")

    for data in data_objs:
        image_id = data.image_id
        image_path = data.image_path
        car_mask_path = data.car_mask_path
        shadow_mask_path = data.shadow_mask_path
        wall_path = data.wall_path
        floor_path = data.floor_path
        results_path = os.path.join("results", f"{image_id}.png")

        image_orig = load_image(image_path)
        car_mask = load_image(car_mask_path)
        shadow_mask = load_image(shadow_mask_path)
        wall_image = load_image(wall_path)
        floor_image = load_image(floor_path)

        final_image = replace_background(
            image_orig, car_mask, shadow_mask, wall_image, floor_image
        )
        save_image(results_path, final_image)
