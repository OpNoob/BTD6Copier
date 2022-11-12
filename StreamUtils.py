import json
import math
import os
import threading
import time
# import pickle
import dill as pickle

import static
from Utils import Utils
from Events import *

import cap_from_youtube
import cv2
import numpy as np
from tqdm import tqdm


class StreamUtils(Utils):
    def __init__(self, thresh=200, search_mouse_pixels=200):
        Utils.__init__(self)

        # Static variables
        self.thresh = thresh
        self.search_mouse_pixels = search_mouse_pixels

        # Set images
        self._mouse_normal_template = self._load_image("mouse 1080p", image_path=True)
        self._mouse_normal_mask = self._connectLines(self._getOutline(self._mouse_normal_template.copy()), fill=True)
        self._mouse_click_template = self._load_image("mouse click 1080p", image_path=True)
        self._mouse_click_mask = self._connectLines(self._getOutline(self._mouse_click_template.copy()), fill=True)

        self._play_template = self._load_image("play", image_path=True)
        self._no_fast_forward_template = self._load_image("no fast forward", image_path=True)
        self._fast_forward_template = self._load_image("fast forward", image_path=True)
        self._fast_forward_template = self._load_image("fast forward", image_path=True)
        self._victory_template = self._load_image("victory", image_path=True)

        self._monkey_grid_enabled_templates = self._loadMonkeyGrid()
        self._monkey_grid_disabled_templates = self._loadMonkeyGridDisabled()

        # scroll
        self._scroll_pixel_total = 0
        self._limit_scroll = 13
        self._points_previous = None
        self._scroll_previous = 0

        # Mouse
        self._search_area = None
        # Scaling
        self.mouse_normal_template = None
        self.mouse_normal_mask = None
        self.mouse_click_template = None
        self.mouse_click_mask = None
        # Get shapes of templates
        self.mouse_normal_template_w, self.mouse_normal_template_h = None, None
        self.mouse_click_template_w, self.mouse_click_template_h = None, None
        # Searching pixels for mouse (So no need to search in all the image)
        self.search_pixels = None
        # Hold previous mouse event
        self._thresh_mouse_move = 5  # In moves
        self._previous_mouse_point = None

        # Fast forward
        self._fast_forward = None

    def _getOutline(self, img_gray):
        # return img_gray
        img_gray[img_gray > self.thresh] = 255
        img_gray[img_gray <= self.thresh] = 0

        # kernel = np.ones((3, 3), np.uint8)
        # outlines = cv2.dilate(img_gray, kernel, iterations=1)

        return img_gray

    def _connectLines(self, img_bin, fill=False):
        # https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
        def find_if_close(cnt1, cnt2):
            row1, row2 = cnt1.shape[0], cnt2.shape[0]
            for i in range(row1):
                for j in range(row2):
                    dist = np.linalg.norm(cnt1[i] - cnt2[j])
                    if abs(dist) < 50:
                        return True
                    elif i == row1 - 1 and j == row2 - 1:
                        return False

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        LENGTH = len(contours)
        status = np.zeros((LENGTH, 1))
        for i, cnt1 in enumerate(contours):
            x = i
            if i != LENGTH - 1:
                for j, cnt2 in enumerate(contours[i + 1:]):
                    x = x + 1
                    dist = find_if_close(cnt1, cnt2)
                    if dist == True:
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1

        unified = []
        maximum = int(status.max()) + 1
        for i in range(maximum):
            pos = np.where(status == i)[0]
            if pos.size != 0:
                cont = np.vstack([contours[i] for i in pos])
                hull = cv2.convexHull(cont)
                unified.append(hull)

        # cv2.drawContours(img_bin, unified, -1, (0, 255, 0), 2)
        cv2.drawContours(img_bin, unified, -1, 255, -1 if fill else 1)

        # kernel = np.ones((2, 2), np.uint8)
        # img_bin = cv2.erode(img_bin, kernel, iterations=1)

        return img_bin

    def _scaleStatic(self, area: tuple[tuple, tuple], integer=True):
        scaled_area = ((self.width * area[0][0]), self.height * area[0][1]), (
            self.width * area[1][0], self.height * area[1][1])
        if integer:
            scaled_area = (
                (round(scaled_area[0][0]), round(scaled_area[0][1])),
                (round(scaled_area[1][0]), round(scaled_area[1][1])))
        return scaled_area

    def _getImageSearch(self, image, norm_area: tuple[tuple, tuple]):
        area = self._scaleStatic(norm_area)
        image_search = image[area[0][1]: area[1][1], area[0][0]:area[1][0]]
        return image_search

    def check_play(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]), self._play_template)

    def check_victory(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["victory"]), self._victory_template)

    def check_fast_forward(self, image_gray, image_write=None):
        search_location = static.search_locations["play"]
        return self._find(self._getImageSearch(image_gray, search_location),
                          self._fast_forward_template,
                          image_write=self._getImageSearch(image_write, search_location))

    def check_no_fast_forward(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]),
                          self._no_fast_forward_template)

    def _loadMonkeyGrid(self, folder="Monkey Grid/enabled", limit=22):
        grid_dict = dict()
        directory = os.path.join(self.support_dir, folder)

        for index in range(limit):
            file_name = f"{index}.png"
            file_path = os.path.join(directory, file_name)
            if os.path.exists(file_path):
                grid_dict[index] = self._load_image(file_path)
            else:
                break
        return grid_dict

    def _loadMonkeyGridDisabled(self, folder="Monkey Grid/disabled"):
        return self._loadMonkeyGrid(folder=folder)

    def _find_grid(self, image_gray, image_write=None):
        monkey_grid = self._scaleStatic(static.monkey_grid)
        if image_write is not None:
            cv2.rectangle(image_write, monkey_grid[0], monkey_grid[1], (255, 255, 255), 2)

        found_indexes = list()
        for index, image_template in self._monkey_grid_enabled_templates.items():
            cv2.imshow("image_template", image_template)
            if self._find(image_gray, image_template):
                found_indexes.append(index)

    def _locate_multi(self, index, return_dict, image_grid_gray, image_template):
        area = self._locate(image_grid_gray, image_template)
        if area is not None:
            return_dict[index] = area

    def _find_scroll(self, image_gray, image_write=None):
        monkey_grid = self._scaleStatic(static.monkey_grid)
        image_grid_gray = self._getImageSearch(image_gray, static.monkey_grid)
        if image_write is not None:
            cv2.rectangle(image_write, monkey_grid[0], monkey_grid[1], (255, 255, 255), 2)

        found_points = dict()
        # Iterative processing
        # for index, image_template in self._monkey_grid_enabled_templates.items():
        #     area = self._locate(image_grid_gray, image_template)
        #     if area is not None:
        #         found_points[index] = area

        # Try multithreding
        threads = []
        for index, image_template in self._monkey_grid_enabled_templates.items():
            t = threading.Thread(target=self._locate_multi, args=(index, found_points, image_grid_gray, image_template))
            threads.append(t)
        for x in threads:
            x.start()
        for x in threads:
            x.join()

        return found_points

    def setMouseScaling(self):
        # Scaling
        self.mouse_normal_template = self.scale(
            self._mouse_normal_template.copy())
        self.mouse_normal_mask = self.scale(self._mouse_normal_mask.copy())
        self.mouse_click_template = self.scale(self._mouse_click_template.copy())
        self.mouse_click_mask = self.scale(self._mouse_click_mask.copy())

        # Get shapes of templates
        self.mouse_normal_template_w, self.mouse_normal_template_h = self.mouse_normal_template.shape[::-1]
        self.mouse_click_template_w, self.mouse_click_template_h = self.mouse_click_template.shape[::-1]

        # Searching pixels for mouse (So no need to search in all the image)
        self.search_pixels = round(self.search_mouse_pixels * self.height / static.base_resolution_height)

        # return MouseEvent()

    def getMouseEvent(self, frame_gray, frame_write=None):
        if self._search_area is None:
            frame_search = frame_gray
        else:
            frame_search = frame_gray[self._search_area[0][1]:self._search_area[1][1],
                           self._search_area[0][0]:self._search_area[1][0]]
            if frame_write is not None:
                frame_write = frame_write[self._search_area[0][1]:self._search_area[1][1],
                              self._search_area[0][0]:self._search_area[1][0]]

        normal_max, normal_pos = self._locate_max(frame_search, self.mouse_normal_template,
                                                  image_mask=self.mouse_normal_mask,
                                                  image_write=frame_write, rect_color=None, draw_border=True,
                                                  return_center=False)
        click_max, click_pos = self._locate_max(frame_search, self.mouse_click_template,
                                                image_mask=self.mouse_click_mask,
                                                image_write=frame_write, rect_color=None, draw_border=True,
                                                return_center=False)

        if normal_max >= click_max:
            point = normal_pos
            templateHeight, templateWidth = self.mouse_normal_template.shape[:2]
            rect_color = (255, 0, 0)
            click = False
        else:
            point = click_pos
            templateHeight, templateWidth = self.mouse_click_template.shape[:2]
            rect_color = (0, 0, 255)
            click = True

        point_center = self._getCenter(point[0], point[1], templateWidth, templateHeight)
        cv2.circle(frame_write, point_center, radius=2, color=(0, 0, 0), thickness=-1)
        if frame_write is not None:
            cv2.rectangle(frame_write, point, (point[0] + templateWidth, point[1] + templateHeight), rect_color, 2)

        x_add = 0
        y_add = 0
        if self._search_area is not None:
            x_add += self._search_area[0][0]
            y_add += self._search_area[0][1]
        self._search_area = ((point[0] - self.search_pixels + x_add, point[1] - self.search_pixels + y_add), (
            point[0] + self.mouse_click_template_w + self.search_pixels + x_add,
            point[1] + self.mouse_click_template_h + self.search_pixels + y_add))

        # Making sure search area not out of bound (might not need this)
        if self._search_area is not None:
            if self._search_area[0][0] < 0:
                self._search_area = ((0, self._search_area[0][1]), (self._search_area[1][0], self._search_area[1][1]))
            if self._search_area[0][1] < 0:
                self._search_area = ((self._search_area[0][0], 0), (self._search_area[1][0], self._search_area[1][1]))
            if self._search_area[1][0] >= frame_gray.shape[1]:
                self._search_area = (
                    (self._search_area[0][0], self._search_area[0][1]),
                    (frame_gray.shape[1] - 1, self._search_area[1][1]))
            if self._search_area[1][1] >= frame_gray.shape[0]:
                self._search_area = (
                    (self._search_area[0][0], self._search_area[0][1]),
                    (self._search_area[1][0], frame_gray.shape[0] - 1))

        # Get point on game screen
        point_on_game = (point[0] + self._search_area[0][0], point[1] + self._search_area[0][1])
        point_on_game_normalised = self.normPoint(point_on_game[0], point_on_game[1])
        # Check if new mouse event
        if self._previous_mouse_point is None or self._thresh_mouse_move < math.dist(self._previous_mouse_point, point_on_game):
            self._previous_mouse_point = point_on_game
            return MouseEvent(point_on_game_normalised[0], point_on_game_normalised[1], click=click)

    def getFastForwardEvent(self, image_gray, image_write=None):
        new_ff = self.check_fast_forward(image_gray, image_write=image_write)
        # Only mark changes
        if self._fast_forward is not new_ff:
            self._fast_forward = new_ff
            return FastForwardEvent(new_ff)

    def getScrollEvent(self, image_gray, image_write=None):
        scroll_add = 0

        points = self._find_scroll(image_gray, image_write=image_write)
        if self._points_previous is not None:
            y_shift = 0
            match_count = 0
            for index, (found_x, found_y, _, _) in points.items():
                if index in self._points_previous:
                    (previous_x, previous_y, _, _) = self._points_previous[index]
                    y_shift += previous_y - found_y
                    match_count += 1
            if match_count != 0:
                y_shift /= match_count

            single_scroll_pixel_shift = static.pixels_scroll * self.height

            if y_shift > single_scroll_pixel_shift / 2:
                self._scroll_pixel_total += y_shift

            num_scrolls = round(self._scroll_pixel_total / single_scroll_pixel_shift)
            if num_scrolls > self._limit_scroll:
                num_scrolls = self._limit_scroll
            scroll_add = num_scrolls - self._scroll_previous
            self._scroll_previous = num_scrolls
            # print("scroll_add: ", scroll_add)
        self._points_previous = points

        if scroll_add > 0:
            return ScrollEvent(scroll_add * -1)

    def getEvents(self, events: Events, time_stamp, image_gray, image_write=None):
        # Fast-forward check
        fast_forward_event = self.getFastForwardEvent(image_gray, image_write=image_write)
        events.addEvent(time_stamp, fast_forward_event)

        # Calculate scroll
        scroll_event = self.getScrollEvent(image_gray, image_write=image_write)
        events.addEvent(time_stamp, scroll_event)

        # Mouse clicks
        mouse_event = self.getMouseEvent(image_gray, frame_write=image_write)
        events.addEvent(time_stamp, mouse_event)


class ReadYoutube:
    def __init__(self, youtube_url, stream_utils: StreamUtils = None,
                 resolution="1080p60",
                 load_from_pickle=None,
                 save_pickle_path=None, save_json_path=None, periodic_saves=False):
        if load_from_pickle is not None:
            if self.load(load_from_pickle):
                return

        # Default initialisation

        if stream_utils is None:  # If class not inputted, then create new instance
            self.stream_utils = StreamUtils()
        else:
            self.stream_utils = stream_utils

        self.youtube_url = youtube_url
        self.events = Events()

        # Flags and analysis variables
        self.frame_no = None
        self.finished = False
        self.resolution = resolution

        self.periodic_saves = periodic_saves
        self._save_pickle_path = save_pickle_path
        self._save_json_path = save_json_path

        self.time_start = None
        self.time_elapsed = None

    def load(self, path):
        if not os.path.exists(path):
            print("Pickle path not found, starting a new instance")
            return False
        else:
            with open(path, "rb") as f:
                self.__dict__ = pickle.load(f).__dict__
            return True

    def save(self, pickle_path=None, json_path=None):
        self._setTime()

        if pickle_path is None and self._save_pickle_path is not None:
            pickle_path = self._save_pickle_path
        if json_path is None and self._save_json_path is not None:
            json_path = self._save_json_path

        if pickle_path is not None:
            with open(pickle_path, "wb") as f:
                pickle.dump(self, f)
        if json_path is not None:
            with open(json_path, "w") as f:
                json.dump(self.toDict(), f, indent=4)

    def toDict(self):
        return {
            "youtube_url": self.youtube_url,
            "time_elapsed": self.time_elapsed,
            "frame_no": self.frame_no,
            "finished": self.finished,
            "resolution": self.resolution,
            "events": self.events.toDict(),
        }

    def _setTime(self):
        self.time_elapsed = time.time() - self.time_start

    def analyse(self, show_frames=True, rescale_show_frames=(960, 540)):
        self.time_start = time.time()

        # Start stream to extract height and width
        streams, resolutions = cap_from_youtube.list_video_streams(self.youtube_url)
        if self.resolution in resolutions:
            res_index = np.where(resolutions == self.resolution)[0][0]
            stream_url = streams[res_index].url
            cap = cv2.VideoCapture(stream_url)
        else:
            raise ValueError(f'Resolution {self.resolution} not available')

        # Skip to desired testing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 8700)

        self.stream_utils.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.stream_utils.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.stream_utils.setMouseScaling()

        # Check if started
        started = True

        fps = cap.get(cv2.CAP_PROP_FPS)

        # infinite loop
        while True:
            ret, frame = cap.read()
            self.frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_stamp = self.frame_no / fps  # In seconds

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not started and self.stream_utils.check_play(frame_gray):
                started = True

            if started:
                # Victory check
                if self.stream_utils.check_victory(frame_gray):
                    break

                self.stream_utils.getEvents(self.events, time_stamp, frame_gray, image_write=frame)
                if self.periodic_saves:
                    self.save()

            if show_frames:
                if rescale_show_frames is not None:
                    frame = cv2.resize(frame, rescale_show_frames, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Output Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    su = StreamUtils()
    ry = ReadYoutube("https://www.youtube.com/watch?v=R3XUmq8_8j0", stream_utils=su, periodic_saves=True, save_pickle_path="data/test.pkl", save_json_path="data/test.json")
    ry.analyse()
