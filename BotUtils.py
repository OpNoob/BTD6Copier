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
from Utils import Utils
from Events import *
from StreamUtils import ReadYoutube


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


class BotUtils(Utils):
    def __init__(self):
        Utils.__init__(self)
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

            # Mouse
            self._previous_click = None

            # Fast forward
            self._previous_fast_forward = None
        except Exception as e:
            raise Exception("Could not retrieve monitor resolution")

    def _move_mouse(self, location, move_timeout=0.1):
        mouse.move(x=location[0], y=location[1])
        time.sleep(move_timeout)

    def _scroll_mouse(self, num_times):
        if num_times < 0:
            wheel_dir = -1
        elif num_times > 0:
            wheel_dir = 1
        else:
            return
        for i in range(abs(num_times)):
            mouse.wheel(wheel_dir)

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
        return w, h

    def _scalePoint(self, x, y, offset_left_top=False):
        x *= self.width
        y *= self.height

        if offset_left_top:
            y += self.top
            x += self.left

        return x, y

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

    def handleEvent(self, event: Event):
        if isinstance(event, MouseEvent):
            if event.click is not self._previous_click:
                if event.click:
                    mouse.press(button='left')
                elif self._previous_click is not None:
                    mouse.release(button='left')
                self._previous_click = event.click
            self._move_mouse(self._scaling(event.x, event.y, offset_left_top=True), 0)

            # self._move_mouse((event.x + self.left, event.y + self.top))
        elif isinstance(event, ScrollEvent):
            # print("scroll: ", event.num_times)
            self._scroll_mouse(event.num_times)
        else:
            pass

    def inputEvents(self, events: dict):
        total_test = 0

        time_stamp_last = 0
        for ts, events in events.items():
            sleep_duration = ts - time_stamp_last
            time_stamp_last = ts
            time.sleep(sleep_duration)

            # total_test += sleep_duration
            # print(total_test, sleep_duration)

            for e in events:
                print(ts, e.toDict())
                self.handleEvent(e)


if __name__ == "__main__":
    delay = 5
    print(f"Sleeping for {delay} seconds")
    time.sleep(delay)
    print("Starting...")
    inst = BotUtils()
    ry = ReadYoutube(None, load_from_pickle="data/test.pkl")
    inst.inputEvents(ry.events.sort())
