import cv2
import os


class MonkeyImage:
    def __init__(self, image_data, top: bool, x=None, y=None):
        self.image_data = image_data
        self.top = top
        self.x = x
        self.y = y

    def getPos(self, round_int=True):
        if round_int:
            return round(self.x), round(self.y)
        return self.x, self.y

    def toDict(self):
        return {
            "top": self.top,
            "x": self.x,
            "y": self.y,
        }


class MonkeyGrid:

    def __init__(self, base_dir):
        self.base_dir = base_dir

        self.enabled_folder = "enabled"
        self.disabled_folder = "disabled"
        self.top_folder = "top"
        self.bottom_folder = "bottom"

    class ImageGrid:
        def __init__(self, image_path: str, enabled: bool, top: bool, index_skips: list[int] = None):
            self.image_path = image_path
            self.enabled = enabled
            self.top = top
            self.index_skips = index_skips if index_skips is not None else list()

    def loadImages(self, scale_to_shape=None):
        return self.extractImages(write_images=False, show_images=False, grayscale=True, scale_to_shape=scale_to_shape)

    def extractImages(self, write_images=True, show_images=False, grayscale=True, scale_to_shape=None):
        monkey_list_enabled = list()
        monkey_list_disabled = list()

        if write_images:
            for folder_1 in [self.enabled_folder, self.disabled_folder]:
                for folder_2 in [self.top_folder, self.bottom_folder]:
                    folder_path = os.path.join(self.base_dir, folder_1, folder_2)
                    os.makedirs(folder_path, exist_ok=True)

        image_grids = [
            self.ImageGrid(os.path.join(self.base_dir, "monkey grid enabled 1.png"), True, True, index_skips=[0]),
            self.ImageGrid(os.path.join(self.base_dir, "monkey grid enabled 2.png"), True, False, index_skips=[11]),
            self.ImageGrid(os.path.join(self.base_dir, "monkey grid disabled 1.png"), False, True, index_skips=[0]),
            self.ImageGrid(os.path.join(self.base_dir, "monkey grid disabled 2.png"), False, False, index_skips=[11]),
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
            if grayscale:
                image_grid = cv2.imread(ig.image_path, 0)
            else:
                image_grid = cv2.imread(ig.image_path)
            image_grid_copy = image_grid.copy() if show_images else None

            h, w = image_grid.shape[:2]
            width_segments = round(w / 2)
            width_shift = round(w / (2 * 2))
            verticals = [i * width_segments + width_shift for i in range(2)]

            height_segments = round(h / 6)
            height_shift = round(h / (6 * 2))
            horizontals = [i * height_segments + height_shift for i in range(6)]

            # Drawing lines
            if show_images:
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

                        if show_images:
                            # Drawing roi
                            cv2.rectangle(image_grid_copy, (vertical + top_bottom_pad, horizontal + left_right_pad),
                                          (vertical - top_bottom_pad, horizontal - left_right_pad), [0, 0, 255], 2)
                            # Showing roi
                            cv2.imshow("roi", roi)

                        # Get folder
                        folder_path = self.base_dir
                        if ig.enabled:
                            index_name = counter_enabled
                            counter_enabled += 1
                            folder_path = os.path.join(folder_path, self.enabled_folder)
                        else:
                            index_name = counter_disabled
                            counter_disabled += 1
                            folder_path = os.path.join(folder_path, self.disabled_folder)
                        if ig.top:
                            folder_path = os.path.join(folder_path, self.top_folder)
                        else:
                            folder_path = os.path.join(folder_path, self.bottom_folder)

                        if write_images:
                            cv2.imwrite(os.path.join(folder_path, f"{index_name}.png"), roi)

                        if scale_to_shape is None:
                            x = vertical
                            y = horizontal
                        else:
                            scale_h, scale_w = scale_to_shape[:2]
                            norm_x = vertical / w
                            norm_y = horizontal / h
                            x = norm_x * scale_w
                            y = norm_y * scale_h

                        if ig.enabled:
                            monkey_list_enabled.append(MonkeyImage(roi, ig.top, x, y))
                        else:
                            monkey_list_disabled.append(MonkeyImage(roi, ig.top, x, y))

                    counter_local += 1  # Counter at the end

            if show_images:
                # Showing image
                cv2.imshow("image_grid_copy", image_grid_copy)

        return monkey_list_enabled, monkey_list_disabled


if __name__ == "__main__":
    mg = MonkeyGrid("assets\\Monkey Grid")
    mg.extractImages()
