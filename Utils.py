import os
from pathlib import Path
import cv2
import numpy as np
import sys
import pytesseract
import re

import static

if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class Utils:
    def __init__(self):
        self.width = None
        self.height = None

        self.support_dir = self.get_resource_dir("assets")
        self._image_path = lambda image, root_dir=self.support_dir: root_dir / f"{image}.png"

        self._last_round_image = None
        self._previous_round = None

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

        result = cv2.matchTemplate(image, template_image,
                                   cv2.TM_CCOEFF_NORMED)  # heatmap of the template and the screenshot"
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

    def _getLocation(self, square):
        top = square[0][1]
        left = square[0][0]
        width = square[1][0] - square[0][0]
        height = square[1][1] - square[0][1]

        return top, left, width, height

    def _locate(self, image, template_image, confidence: float = 0.9,
                image_write=None, draw_image_write=False):
        """
            Locates a template on the screen.

            Note: @tries does not do anything at the moment
        """
        result = self._locate_all(image, template_image, confidence)

        if draw_image_write and image_write is not None:
            h, w = image_write.shape[:2]
            cv2.rectangle(image_write, (1, 1), (w - 1, h - 1), (0, 0, 0), 2)

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
                cv2.rectangle(image_write, max_pos, (max_pos[0] + templateWidth, max_pos[1] + templateHeight),
                              rect_color, 2)

            if draw_border:
                h, w = image_write.shape[:2]
                cv2.rectangle(image_write, (1, 1), (w - 1, h - 1), (0, 0, 0), 2)

        if return_center:
            return max_res, self._getCenter(max_pos[0], max_pos[1], templateWidth, templateHeight)
        else:
            return max_res, max_pos

    def _scaleStatic(self, area: tuple[tuple, tuple], integer=True):
        scaled_area = ((self.width * area[0][0]), self.height * area[0][1]), (
            self.width * area[1][0], self.height * area[1][1])
        if integer:
            scaled_area = (
                (round(scaled_area[0][0]), round(scaled_area[0][1])),
                (round(scaled_area[1][0]), round(scaled_area[1][1])))
        return scaled_area

    def _getRound(self, img_bgr, image_write=None, show_image=False, thresh_change=0.01):  # 0.0001
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 245]), np.array([255, 255, 255]))  # 240

        if show_image:
            cv2.imshow("white_mask", white_mask)

        round_num = None

        # Skip ocr if same image
        ratio_diff = self.getImageDiff(self._last_round_image, white_mask)
        if ratio_diff > thresh_change:
            self._last_round_image = white_mask

            found_text = pytesseract.image_to_string(white_mask,
                                                     config='--psm 7 -c tessedit_char_whitelist=0123456789/').replace(
                "\n", "")

            # print(f"found_text '{found_text}'")
            if re.search(r"(\d+/\d+)", found_text):
                found_text = re.search(r"(\d+)", found_text)
                round_num = int(found_text.group(0))
                self._previous_round = round_num

            # Ensures that a number is retrieved if nothing was gotten
            if round_num is None:
                self._last_round_image = None

        if image_write is not None:
            if round_num is not None:
                text_color = (255, 0, 0)
            elif ratio_diff > thresh_change:
                text_color = (0, 255, 0)
            else:
                text_color = (0, 0, 255)

            text = str(self._previous_round)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1
            font_weight = 4
            textsize = cv2.getTextSize(text, font, font_size, font_weight)[0]
            textX = round((image_write.shape[1] - textsize[0]) / 2)
            textY = round((image_write.shape[0] + textsize[1]) / 2)
            cv2.putText(image_write, text, (textX, textY), font, font_size, text_color, font_weight)

        return round_num

    def getImageDiff(self, img1, img2, show_image=False):
        if img1 is None or img2 is None:
            return 1.0

        res = cv2.absdiff(img1, img2)
        res = res.astype(np.uint8)

        ratio = (np.count_nonzero(res)) / res.size

        if show_image:
            cv2.imshow("diff res", res)
            print("ratio: ", ratio)

        return ratio
