import os

import cv2
import numpy as np


class Data:
    def __init__(self, data_folder, image_id):
        image_paths = os.path.join(data_folder, "images")
        car_mask_paths = os.path.join(data_folder, "car_masks")
        shadow_mask_paths = os.path.join(data_folder, "shadow_masks")

        self.image_id = image_id
        self.image_path = os.path.join(image_paths, f"{image_id}.jpeg")
        self.car_mask_path = os.path.join(car_mask_paths, f"{image_id}.png")
        self.shadow_mask_path = os.path.join(shadow_mask_paths, f"{image_id}.png")
        self.wall_path = os.path.join(data_folder, "wall.png")
        self.floor_path = os.path.join(data_folder, "floor.png")


def load_image(path):
    """Load image from path"""
    return cv2.imread(path)


def save_image(path, image):
    """Save image to path"""
    cv2.imwrite(path, image)


def remove_noise(image):
    """Remove noise from image
    Uses median blur to remove noise
    """
    image = cv2.medianBlur(image, 3)
    return image


def fill_holes(image):
    """Fill holes in images
    Uses morphology to fill holes
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    processed_image = cv2.morphologyEx(inverted_gray, cv2.MORPH_OPEN, kernel)
    return processed_image


def make_custom_bg(input_image, wall_image, floor_image):
    """Make custom background.
    Crops wall and floor from the center and concatenates them
    to form the custom background.
    Using cropping to avoid resizing the background.
    """
    height, width, _ = input_image.shape
    aspect_ratio = width / height

    # Wall and floor dimensions
    wall_height = 900
    floor_height = 2160 - wall_height  # 2160 is original floor image height
    wall_width = int(aspect_ratio * 2160)  # Getting the width of wall
    floor_width = int(aspect_ratio * 2160)  # Getting the width of floor

    # Cropping wall and floor from the center
    # This is done to make sure that there would be no need for resizing bg
    # Wall array pos
    wall_startx = (3840 - wall_width) // 2
    wall_endx = wall_startx + wall_width
    wall_starty = 600
    wall_endy = wall_starty + wall_height

    # Floor array pos
    floor_startx = (3840 - floor_width) // 2
    floor_endx = floor_startx + floor_width
    floor_starty = 1260
    floor_endy = floor_starty + floor_height

    # Cropping wall and floor
    wall = wall_image[wall_starty:wall_endy, wall_startx:wall_endx, :]
    floor = floor_image[floor_starty:floor_endy, floor_startx:floor_endx, :]

    # Concatenating wall and floor
    # To form the custom background
    custom_bg = np.concatenate([wall, floor])
    custom_bg = cv2.resize(custom_bg, (width, height))

    return custom_bg


def get_car_bbox(input_image, mask):
    """Get bounding box of car in input image"""
    car_mask_processed = mask.copy()
    input_height, input_width = input_image.shape[:2]

    # Finding the bounding box of car
    x, y, w, h = cv2.boundingRect(255 - car_mask_processed)

    # Defining the padding
    x_padding = 10
    y_padding = 80
    if h < input_height / 2:
        y_padding = 160

    # Calculating the start and end points of bounding box
    startx = x - x_padding
    starty = y - 20
    width = w + 2 * x_padding
    height = h + y_padding
    return [startx, starty, width, height]


def place_car_in_custom_bg(input_image, output_image, mask):
    """Place car in custom background"""
    h1, w1 = output_image.shape[:2]

    # Getting the car masks for input and result
    bg_result_mask = np.zeros((h1, w1, 3), dtype=np.bool)
    bg_input_mask = mask == 255
    bg_input_mask = np.broadcast_to(bg_input_mask[:, :, np.newaxis], input_image.shape)

    car_bbox = get_car_bbox(input_image, mask)
    x_old, y_old, w, h = car_bbox

    x_new = (w1 - w) // 2
    y_new = h1 - h
    bg_result_mask[y_new:, x_new : x_new + w, :] = ~bg_input_mask[
        y_old : y_old + h, x_old : x_old + w, :
    ]

    output_image[bg_result_mask] = input_image[~bg_input_mask]
    return output_image, bg_result_mask


def place_shadow_in_custom_bg(shadow_mask, result_image, bg_result_mask):
    """Place shadow in custom background.
    Uses template matching to place shadow in custom background
    """
    # Get shadow template
    shadow_template = shadow_mask.copy()
    shadow_template[shadow_mask != 0] = 255

    # Match shadow template to get shadow position
    image_without_bg = result_image.copy()
    image_without_bg[~bg_result_mask] = 0
    result = cv2.matchTemplate(image_without_bg, shadow_template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = list(min_loc)
    h, w, _ = shadow_template.shape

    new_sh_mask = shadow_mask.copy()
    new_sh_mask = cv2.dilate(new_sh_mask, np.ones((7, 7), np.uint8), iterations=5)
    new_sh_mask = np.clip(new_sh_mask, 0, 120)
    return new_sh_mask, top_left


def replace_background(image_orig, car_mask, shadow_mask, wall_image, floor_image):
    """Replace background of image with custom background"""
    result_image = np.zeros_like(image_orig)
    car_mask_processed = fill_holes(remove_noise(car_mask))
    bg_mask = car_mask_processed == 255

    bg = make_custom_bg(image_orig, wall_image, floor_image)
    result_image, bg_result_mask = place_car_in_custom_bg(
        image_orig, result_image, car_mask_processed
    )

    result_image = bg.copy()
    orig_mask = np.broadcast_to(bg_mask[:, :, np.newaxis], image_orig.shape)
    result_image[bg_result_mask] = image_orig[~orig_mask]

    new_sh_mask, top_left = place_shadow_in_custom_bg(
        shadow_mask, result_image, bg_result_mask
    )
    h, w, _ = shadow_mask.shape
    result_image = bg.copy()

    result_image[
        top_left[1] : top_left[1] + h, top_left[0] : top_left[0] + w, :
    ] -= new_sh_mask

    result_image[bg_result_mask] = image_orig[~orig_mask]
    final_image = np.clip(result_image, 0, 255)
    return final_image
