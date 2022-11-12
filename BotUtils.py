import sys
import time
import keyboard
import mouse
import static
import tkinter
from pathlib import Path

import numpy as np
import cv2

import re
import mss

import ctypes
from collections import defaultdict

import math
from win32gui import FindWindow, GetClientRect, GetWindowRect


class SearchLocation:
    def __init__(self, top, left, bottom, right):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def getWidth(self):
        return self.right - self.left

    def getHeight(self):
        return self.bottom - self.top


class BotUtils:
    def __init__(self):

        try:
            if sys.platform == "win32":
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # DPI indipendent
            # tk = tkinter.Tk()
            # self.width, self.height = tk.winfo_screenwidth(), tk.winfo_screenheight()

            self.window_handle = FindWindow(None, static.window_title)

            # https://stackoverflow.com/questions/51287338/python-2-7-get-ui-title-bar-size
            window_rect = GetWindowRect(self.window_handle)
            client_rect = GetClientRect(self.window_handle)
            windowOffset = math.floor(((window_rect[2] - window_rect[0]) - client_rect[2]) / 2)
            titleOffset = ((window_rect[3] - window_rect[1]) - client_rect[3]) - windowOffset
            game_rect = (window_rect[0] + windowOffset, window_rect[1] + titleOffset, window_rect[2] - windowOffset,
                         window_rect[3] - windowOffset)

            self.left, self.top, self.right, self.bottom = game_rect
            self.width = self.right - self.left
            self.height = self.bottom - self.top
        except Exception as e:
            raise Exception("Could not retrieve monitor resolution")

        self.support_dir = self.get_resource_dir("assets")

        # Defing a lamda function that can be used to get a path to a specific image
        # self._image_path = lambda image, root_dir=self.support_dir, height=self.height : root_dir/f"{height}_{image}.png"
        self._image_path = lambda image, root_dir=self.support_dir: root_dir / f"{image}.png"

    def get_resource_dir(self, path):
        return Path(__file__).resolve().parent / path

    def save_file(self, data=format(0, 'b'), _file_name="noname", folder="DEBUG", ):
        directory = Path(__file__).resolve().parent / folder

        if not directory.exists():
            Path.mkdir(directory)

        with open(directory / _file_name, "wb") as output_file:
            output_file.write(data)

    def _move_mouse(self, location, move_timeout=0.1):
        mouse.move(x=location[0], y=location[1])
        time.sleep(move_timeout)

    def click(self, location: tuple | tuple, amount=1, timeout=0.5, move_timeout=0.1, press_time=0.075):
        """
            Method to click on a specific location on the screen
            @param location: The location to click on
            @param amount: amount of clicks to be performed
            @param timeout: time to wait between clicks
            @param move_timeout: time to wait between move and click
            @param press_time: time to wait between press and release
        """

        # If location is a string then assume that its a static button
        if isinstance(location, str):
            location = static.button_positions[location]

        # Move mouse to location
        self._move_mouse(self._scaling(location, offset_left_top=True), move_timeout)

        for _ in range(amount):
            mouse.press(button='left')
            time.sleep(
                press_time)  # https://www.reddit.com/r/AskTechnology/comments/4ne2tv/how_long_does_a_mouse_click_last/ TLDR; dont click too fast otherwise shit will break
            mouse.release(button='left')

            """
                We don't need to apply cooldown and slow down the bot on single clicks
                So we only apply the .1 delay if the bot has to click on the same spot multiple times
                This is currently used for level selection and levelup screen
            """
            if amount > 1:
                time.sleep(timeout)

        time.sleep(timeout)

    def press_key(self, key, timeout=0.1, amount=1):
        for _ in range(amount):
            keyboard.send(key)
            time.sleep(timeout)

    # Scaling functions for different resolutions support
    def _scaling(self, width, height, offset_left_top=False):
        resize_h = self.height / static.base_resolution_height
        resize_w = self.width / static.base_resolution_width
        h = height * resize_h
        w = width * resize_w
        if offset_left_top:
            h += self.top
            w += self.left
        return width, height

    def _load_img(self, img):

        if isinstance(img, Path):
            # The function imread loads an image from the specified file and
            # returns it. If the image cannot be read (because of missing
            # file, improper permissions, unsupported or invalid format),
            # the function returns an empty matrix
            # http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
            img_cv = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                raise IOError(
                    f"Failed to read {img} because file is missing, has improper permissions, or is an unsupported or invalid format")
        elif isinstance(img, np.ndarray):
            img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # don't try to convert an already-gray image to gray
            # if grayscale and len(img.shape) == 3:  # and img.shape[2] == 3:
            # else:
            #     img_cv = img
        elif hasattr(img, 'convert'):
            # assume its a PIL.Image, convert to cv format
            img_array = np.array(img.convert('RGB'))
            img_cv = img_array[:, :, ::-1].copy()  # -1 does RGB -> BGR
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError('expected an image filename, OpenCV numpy array, or PIL image')

        return img_cv

    def _locate_all(self, template_path, search_location: SearchLocation, confidence=0.9, limit=100, region=None):
        """
            Template matching a method to match a template to a screenshot taken with mss.

            @template_path - Path to the template image
            @confidence - A threshold value between {> 0.0f & < 1.0f} (Defaults to 0.9f)

            credit: https://github.com/asweigart/pyscreeze/blob/b693ca9b2c964988a7e924a52f73e15db38511a8/pyscreeze/__init__.py#L184

            Returns a list of cordinates to where openCV found matches of the template on the screenshot taken
        """

        monitor = {'top': search_location.top, 'left': search_location.left, 'width': search_location.getWidth(),
                   'height': search_location.getHeight()} if region is None else region

        if 0.0 > confidence <= 1.0:
            raise ValueError("Confidence must be a value between 0.0 and 1.0")

        with mss.mss() as sct:

            # Load the taken screenshot into a opencv img object
            img = np.array(sct.grab(monitor))
            screenshot = self._load_img(img)

            if region:
                screenshot = screenshot[region[1]:region[1] + region[3],
                             region[0]:region[0] + region[2]
                             ]
            else:
                region = (0, 0)
            # Load the template image
            template = self._load_img(template_path)

            confidence = float(confidence)

            # width & height of the template
            templateHeight, templateWidth = template.shape[:2]

            # scale template
            if self.width != 1920 or self.height != 1080:
                template = cv2.resize(template, dsize=(
                    int(templateWidth / (1920 / self.width)), int(templateHeight / (1080 / self.height))),
                                      interpolation=cv2.INTER_CUBIC)

            # Find all the matches
            # https://stackoverflow.com/questions/7670112/finding-a-subimage-inside-a-numpy-image/9253805#9253805
            result = cv2.matchTemplate(screenshot, template,
                                       cv2.TM_CCOEFF_NORMED)  # heatmap of the template and the screenshot"
            match_indices = np.arange(result.size)[(result > confidence).flatten()]
            matches = np.unravel_index(match_indices[:limit], result.shape)

            # Defining the coordinates of the matched region
            matchesX = matches[1] * 1 + region[0]
            matchesY = matches[0] * 1 + region[1]

            if len(matches[0]) == 0:
                return None
            else:
                return [(x + self.left, y + self.top, templateWidth, templateHeight) for x, y in
                        zip(matchesX, matchesY)]

    def _locate(self, template_path, confidence=0.9, tries=1):
        """
            Locates a template on the screen.

            Note: @tries does not do anything at the moment
        """
        result = self._locate_all(template_path, confidence)
        return result[0] if result is not None else None


if __name__ == "__main__":
    import time

    inst = BotUtils()
    inst.log = print
    inst.DEBUG = True
    time.sleep(2)

    print(inst.getRound())

    # res = inst._locate(inst._image_path("obyn"), confidence=0.9)
    # print(res)
