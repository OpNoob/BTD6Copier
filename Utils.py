import os
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

    def normPoint(self, x, y):
        return x / self.width, y / self.height

    def scalePoint(self, x, y):
        return x * self.width, y * self.height

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

    def _matchTemplate(self, image, template_image):
        return cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)

    def _find(self, image, template_image, confidence=0.9, image_write=None):
        if self._locate(image, template_image, confidence=confidence, image_write=image_write) is None:
            found = False
        else:
            found = True

        if image_write is not None:
            h, w = image_write.shape[:2]
            if found:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image_write, (1, 1), (w - 1, h - 1), color, 2)

        return found

    def _locate_all(self, image, template_image, confidence: float = 0.9, limit=100):
        templateHeight, templateWidth = template_image.shape[:2]

        result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)  # heatmap of the template and the screenshot"
        match_indices = np.arange(result.size)[(result > confidence).flatten()]
        matches = np.unravel_index(match_indices[:limit], result.shape)

        # Defining the coordinates of the matched region
        matchesX = matches[1] * 1
        matchesY = matches[0] * 1

        if len(matches[0]) == 0:
            return None
        else:
            return [(x, y, templateWidth, templateHeight) for x, y in zip(matchesX, matchesY)]

    def _getCenter(self, x, y, width, height):
        x_center = x + round(width / 2)
        y_center = y + round(height / 2)

        return x_center, y_center

    def _getSquare(self, location):
        return (location[0], location[1]), (location[0] + location[2], location[1] + location[3])

    def _locate(self, image, template_image, confidence: float = 0.9,
                image_write=None, draw_image_write=False):
        """
            Locates a template on the screen.

            Note: @tries does not do anything at the moment
        """
        result = self._locate_all(image, template_image, confidence)

        if draw_image_write and image_write is not None:
            h, w = image_write.shape[:2]
            cv2.rectangle(image_write, (1, 1), (w-1, h-1), (0, 0, 0), 2)

        if result is None:
            return None
        else:
            location = result[0]

            # if image_write is not None:
            #     cv2.circle(image_write, (location[0], location[1]), radius=2, color=(0, 0, 255), thickness=-1)
            #     square = self._getSquare(location)
            #     cv2.rectangle(image_write, square[0], square[1], (255, 0, 0), 2)

            return location

    def _locate_max(self, image, template_image, image_mask=None,
                    image_write=None, rect_color=(0, 0, 0), draw_border=False,
                    return_center=False):
        if image_write is not None:
            assert image.shape[:2] == image_write.shape[:2]  # Both images must have same shape

        templateHeight, templateWidth = template_image.shape[:2]

        if image_mask is None:
            result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(image, template_image, cv2.TM_CCORR_NORMED, None, image_mask)

        max_res = np.max(result)
        max_pos = np.unravel_index(np.argmax(result, axis=None), result.shape)
        max_pos = tuple(reversed(max_pos))

        if image_write is not None:
            if rect_color is not None:
                cv2.rectangle(image_write, max_pos, (max_pos[0] + templateWidth, max_pos[1] + templateHeight), rect_color, 2)

            if draw_border:
                h, w = image_write.shape[:2]
                cv2.rectangle(image_write, (1, 1), (w-1, h-1), (0, 0, 0), 2)

        if return_center:
            return max_res, self._getCenter(max_pos[0], max_pos[1], templateWidth, templateHeight)
        else:
            return max_res, max_pos
