from pathlib import Path
import cv2
import numpy as np

import static


class Utils:
    def __init__(self):
        self.width = None
        self.height = None

        self.support_dir = self.get_resource_dir("assets")
        self._image_path = lambda image, root_dir=self.support_dir: root_dir / f"{image}.png"

    def get_resource_dir(self, path):
        return Path(__file__).resolve().parent / path

    def save_file(self, data=format(0, 'b'), _file_name="noname", folder="DEBUG", ):
        directory = Path(__file__).resolve().parent / folder

        if not directory.exists():
            Path.mkdir(directory)

        with open(directory / _file_name, "wb") as output_file:
            output_file.write(data)

    def _load_image(self, path, image_path=False):
        if image_path:
            path = self._image_path(path)
        path = self.get_resource_dir(path)
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def scale(self, img):
        assert self.width is not None and self.height is not None  # Width and Height must be set
        if self.height != static.base_resolution_height or self.width != static.base_resolution_width:
            resize_h = self.height / static.base_resolution_height
            resize_w = self.width / static.base_resolution_width

            h, w = img.shape

            new_h = h * resize_h
            new_w = w * resize_w
            dim = (round(new_w), round(new_h))
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)  # cv2.INTER_AREA
            return resized
        else:
            return img

    def _find(self, image, template_image, confidence=0.9):
        result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
        if np.amax(result) > confidence:
            return True
        return False

