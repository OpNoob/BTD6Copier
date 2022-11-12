import cv2
import os

base_dir = "assets\\Monkey Grid"

# Creating directories
enabled_folder = os.path.join(base_dir, "enabled")
disabled_folder = os.path.join(base_dir, "disabled")
for x in [enabled_folder, disabled_folder]:
    os.makedirs(x, exist_ok=True)


class ImageGrid:
    def __init__(self, image_name: str, enabled: bool, index_skips: list[int] = None):
        self.image_path = os.path.join(base_dir, image_name + ".png")
        self.enabled = enabled
        self.index_skips = index_skips if index_skips is not None else list()


image_grids = [
    ImageGrid("monkey grid enabled 1", True, index_skips=[0]),
    ImageGrid("monkey grid enabled 2", True, index_skips=[11]),
    ImageGrid("monkey grid disabled 1", False, index_skips=[0]),
    ImageGrid("monkey grid disabled 2", False, index_skips=[11]),
]

# Static variables
resolution_height = 1080
resolution_width = 1920
monkey_width_norm = 0.03375
monkey_height_norm = 0.0633333

# Worked variables
monkey_width = monkey_width_norm * resolution_width
monkey_height = monkey_height_norm * resolution_height

counter_enabled = 0
counter_disabled = 0

for ig in image_grids:
    image_grid = cv2.imread(ig.image_path)
    image_grid_copy = image_grid.copy()

    h, w, _ = image_grid.shape
    width_segments = round(w / 2)
    width_shift = round(w / (2 * 2))
    verticals = [i * width_segments + width_shift for i in range(2)]

    height_segments = round(h / 6)
    height_shift = round(h / (6 * 2))
    horizontals = [i * height_segments + height_shift for i in range(6)]

    # Drawing lines
    for vertical in verticals:
        cv2.line(image_grid_copy, (vertical, 0), (vertical, h), [0, 0, 0], 2)
    for horizontal in horizontals:
        cv2.line(image_grid_copy, (0, horizontal), (w, horizontal), [255, 255, 255], 2)

    counter_local = 0
    for horizontal in horizontals:
        for vertical in verticals:
            if counter_local not in ig.index_skips:
                left_right_pad = round(monkey_width / 2)
                top_bottom_pad = round(monkey_height / 2)
                roi = image_grid[horizontal - left_right_pad: horizontal + left_right_pad,
                      vertical - top_bottom_pad: vertical + top_bottom_pad]

                # Drawing roi
                cv2.rectangle(image_grid_copy, (vertical + top_bottom_pad, horizontal + left_right_pad),
                              (vertical - top_bottom_pad, horizontal - left_right_pad), [0, 0, 255], 2)

                # Showing roi
                cv2.imshow("roi", roi)

                # Save image to directory
                if ig.enabled:
                    cv2.imwrite(os.path.join(enabled_folder, f"{counter_enabled}.png"), roi)
                    counter_enabled += 1
                else:
                    cv2.imwrite(os.path.join(disabled_folder, f"{counter_disabled}.png"), roi)
                    counter_disabled += 1

            counter_local += 1  # Counter at the end

    # Showing image
    cv2.imshow("image_grid_copy", image_grid_copy)
    # cv2.waitKey(0)
