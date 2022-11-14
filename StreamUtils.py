import json
import math
import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
import pickle
# import dill as pickle

import static
from Utils import Utils
from Events import *
from MonkeyGrid import MonkeyImage, MonkeyGrid

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

        self._monkey_grid_enabled_top_templates = self._loadMonkeyGrid("Monkey Grid/enabled/top")
        self._monkey_grid_enabled_bottom_templates = self._loadMonkeyGrid("Monkey Grid/enabled/bottom")
        self._monkey_grid_disabled_top_templates = self._loadMonkeyGrid("Monkey Grid/disabled/top")
        self._monkey_grid_disabled_bottom_templates = self._loadMonkeyGrid("Monkey Grid/disabled/bottom")

        # scroll
        self._scroll_pixel_total = 0
        self._limit_scroll = 13
        self._scroll_previous = 0  # Last scroll shift
        self._monkey_list_enabled_templates, self._monkey_list_disabled_templates = None, None

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
        self._previous_mouse_click = None
        self._current_mouse_position = None

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
                          image_write=self._getImageSearch(image_write,
                                                           search_location) if image_write is not None else None)

    def check_no_fast_forward(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]),
                          self._no_fast_forward_template)

    def _loadMonkeyGrid(self, folder, limit=12):
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

    def _find_grid(self, image_gray, image_write=None):
        monkey_grid = self._scaleStatic(static.monkey_grid)
        if image_write is not None:
            cv2.rectangle(image_write, monkey_grid[0], monkey_grid[1], (255, 255, 255), 2)

        found_indexes = list()
        for index, image_template in self._monkey_grid_enabled_templates.items():
            cv2.imshow("image_template", image_template)
            if self._find(image_gray, image_template):
                found_indexes.append(index)

    def _locate_multi(self, index, return_dict, image_grid_gray, image_template, image_write=None):
        area = self._locate(image_grid_gray, image_template)

        if area is not None:
            point = self._getCenter(area[0], area[1], area[2], area[3])
            return_dict[index] = point

            if image_write is not None:
                square = self._getSquare(area)
                cv2.rectangle(image_write, square[0], square[1], (255, 0, 0), 2)

                text = str(index)
                font = cv2.FONT_HERSHEY_SIMPLEX
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                textX = point[0] - round(textsize[0] / 2)
                textY = point[1] + round(textsize[1] / 2)
                cv2.putText(image_write, text, (textX, textY), font, 1, (255, 0, 0), 2)

    def _find_scroll(self, image_gray, image_write=None):
        # Load images only once
        if self._monkey_list_enabled_templates is None or self._monkey_list_disabled_templates is None:
            self._monkey_list_enabled_templates, self._monkey_list_disabled_templates = MonkeyGrid(
                self.get_resource_dir("assets/Monkey Grid")).loadImages(image_gray.shape)

        if image_write is not None:
            h, w = image_gray.shape[:2]
            cv2.rectangle(image_write, (1, 1), (w-1, h-1), (255, 255, 255), 2)

        found_points = dict()
        # Iterative processing
        # for index, image_template in self._monkey_grid_enabled_templates.items():
        #     area = self._locate(image_grid_gray, image_template)
        #     if area is not None:
        #         found_points[index] = area

        # Try multithreding
        threads = []

        # Loop over template images
        for index, monkey_dat in enumerate(self._monkey_list_enabled_templates):
            image_template = monkey_dat.image_data
            t = threading.Thread(target=self._locate_multi, args=(index, found_points, image_gray, image_template, image_write))
            threads.append(t)

            if image_write is not None:
                point = monkey_dat.getPos()
                cv2.circle(image_write, point, radius=5, color=(0, 0, 255), thickness=-1)

        # Starting threads
        for x in threads:
            x.start()

        # Closing threads
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

    def getMouseEvent(self, frame_gray, frame_write=None, normalise=False):
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

        # shifting point
        point_shifted = static.mouse_normal_shift[0] + point[0], static.mouse_normal_shift[1] + point[1]

        if frame_write is not None:
            # point_center = self._getCenter(point[0], point[1], templateWidth, templateHeight)
            cv2.circle(frame_write, point_shifted, radius=3, color=(0, 0, 0), thickness=-1)
            cv2.rectangle(frame_write, point, (point[0] + templateWidth, point[1] + templateHeight), rect_color, 2)

        # Get point on game screen
        point_on_game = (point_shifted[0] + x_add, point_shifted[1] + y_add)
        self._current_mouse_position = point_on_game
        # Check if new mouse event
        if click or self._previous_mouse_click is not click or (self.check_mouse_in_scroll(point_on_game) and self._thresh_mouse_move < math.dist(self._previous_mouse_point, point_on_game)):
            self._previous_mouse_point = point_on_game
            self._previous_mouse_click = click

            if normalise:
                point_on_game = self.normPoint(point_on_game[0], point_on_game[1])
            return MouseEvent(point_on_game[0], point_on_game[1], click=click)

    def getFastForwardEvent(self, image_gray, image_write=None):
        new_ff = self.check_fast_forward(image_gray, image_write=image_write)
        # Only mark changes
        if self._fast_forward is not new_ff:
            self._fast_forward = new_ff
            return FastForwardEvent(new_ff)

    def check_mouse_in_scroll(self, point):
        area = self._scaleStatic(static.monkey_grid)
        if area[0][0] <= point[0] <= area[1][0] and area[0][1] <= point[1] <= area[1][1]:
            return True
        else:
            return False

    def getScrollEvent(self, image_gray, image_write=None):
        if self.check_mouse_in_scroll(self._current_mouse_position):
            image_gray = self._getImageSearch(image_gray, static.monkey_grid)
            if image_write is not None:
                image_write = self._getImageSearch(image_write, static.monkey_grid)

            points = self._find_scroll(image_gray, image_write=image_write)

            # print("points: ", dict(sorted(points.items())))
            # print("monkey points: ", {i: (x.x, x.y) for i, x in enumerate(self._monkey_list_enabled_templates)})

            single_scroll_pixel_shift = static.pixels_scroll * self.height

            y_shift = 0
            match_count = 0
            for index, (found_x, found_y) in points.items():
                current_monkey_template = self._monkey_list_enabled_templates[index]
                previous_x, previous_y = current_monkey_template.getPos()

                if current_monkey_template.top:
                    y_shift += previous_y - found_y
                else:
                    y_shift += self._limit_scroll * single_scroll_pixel_shift - (found_y - previous_y)
                match_count += 1
            if match_count != 0:
                y_shift /= match_count

            # if y_shift > single_scroll_pixel_shift / 2:
            self._scroll_pixel_total = y_shift

            num_scrolls = round(self._scroll_pixel_total / single_scroll_pixel_shift)
            if num_scrolls > self._limit_scroll:
                num_scrolls = self._limit_scroll
            scroll_add = num_scrolls - self._scroll_previous
            self._scroll_previous = num_scrolls
            # print("scroll_add: ", scroll_add, "            total_scroll: ", num_scrolls, "     total_shift: ", self._scroll_pixel_total)

            if abs(scroll_add) > 0:
                return ScrollEvent(scroll_add * -1)

    def getEvents(self, events: Events, time_stamp, image_gray, pbar=None, image_write=None):
        # # Fast-forward check
        # fast_forward_event = self.getFastForwardEvent(image_gray, image_write=image_write)
        # events.addEvent(time_stamp, fast_forward_event)

        # Mouse clicks
        mouse_event = self.getMouseEvent(image_gray, frame_write=image_write)
        events.addEvent(time_stamp, mouse_event)

        # Calculate scroll
        scroll_event = self.getScrollEvent(image_gray, image_write=image_write)
        events.addEvent(time_stamp, scroll_event)

        if pbar is not None:
            pbar.update(1)


class ReadYoutube:
    def __init__(self, youtube_url,
                 resolution="1080p60",
                 load_from_pickle=None,
                 save_pickle_path=None, save_json_path=None, periodic_saves=60):
        if load_from_pickle is not None:
            if self.load(load_from_pickle):
                return

        # Default initialisation

        self.youtube_url = youtube_url
        self.events = Events()

        # Flags and analysis variables
        self.frame_no = 0
        self.finished = False
        self.resolution = resolution

        self.periodic_saves = periodic_saves
        self._save_pickle_path = save_pickle_path
        self._save_json_path = save_json_path

        self.time_start = None
        self.time_elapsed = None

        self.time_start = None

        self.resetTimer()

    def load(self, path=None):
        if path is None:
            path = self._save_pickle_path
        if not os.path.exists(path):
            print("Pickle path not found, starting a new instance")
            return False
        else:
            with open(path, "rb") as f:
                self.__dict__ = pickle.load(f).__dict__
            return True

    def resetTimer(self):
        self.time_start = time.time()

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

    def analyse(self, stream_utils=None, show_frames=True, rescale_show_frames=(960, 540), start_seconds=None):
        if stream_utils is None:  # If class not inputted, then create new instance
            stream_utils = StreamUtils()
        else:
            stream_utils = stream_utils

        # Start stream to extract height and width
        streams, resolutions = cap_from_youtube.list_video_streams(self.youtube_url)
        if self.resolution in resolutions:
            res_index = np.where(resolutions == self.resolution)[0][0]
            stream_url = streams[res_index].url
            print("stream_url: ", stream_url)
            cap = cv2.VideoCapture(stream_url)
        else:
            raise ValueError(f'Resolution {self.resolution} not available')

        # Skip to desired testing
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stream_utils.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        stream_utils.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        stream_utils.setMouseScaling()

        fps = cap.get(cv2.CAP_PROP_FPS)

        if start_seconds is None:
            start_frame = self.frame_no
        else:
            start_frame = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 8700

        # Check if started
        started = True

        # periodic saves
        last_save = time.time()

        # infinite loop
        pbar = tqdm(total=frame_count, desc="analysing video", initial=int(start_frame))
        while True:
            ret, frame = cap.read()
            self.frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_stamp = self.frame_no / fps  # In seconds

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not started and stream_utils.check_play(frame_gray):
                self.events.addEvent(time_stamp, StartEvent())
                started = True

            if started:
                # Victory check
                if stream_utils.check_victory(frame_gray):
                    break

                stream_utils.getEvents(self.events, time_stamp, frame_gray, image_write=frame if show_frames else None)
                if self.periodic_saves is not None:
                    if time.time() - last_save > self.periodic_saves:
                        last_save = time.time()
                        self.save()

            if show_frames:
                if rescale_show_frames is not None:
                    frame = cv2.resize(frame, rescale_show_frames, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Output Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.save()
                    break
            pbar.update(1)
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        self.save()
        self.finished = True

    def continue_analysing(self, pickle_path=None, stream_utils=None, show_frames=True, rescale_show_frames=(960, 540),
                           start_seconds=None):
        while not self.finished:
            try:
                if pickle_path is None:
                    pickle_path = self._save_pickle_path
                if os.path.exists(pickle_path):
                    print(f"loading from path {pickle_path}")
                    self.load(pickle_path)
                    print(f"Starting from frame no #{self.frame_no}")
                    self.analyse(stream_utils=stream_utils, show_frames=show_frames, rescale_show_frames=rescale_show_frames,
                                 start_seconds=None)
                else:
                    print("Starting new analysis")
                    self.analyse(stream_utils=stream_utils, show_frames=show_frames, rescale_show_frames=rescale_show_frames,
                                 start_seconds=start_seconds)
            except Exception as e:
                print("ERROR! ", str(e))


if __name__ == "__main__":
    su = StreamUtils()
    ry = ReadYoutube("https://www.youtube.com/watch?v=WzRUjzZeoy8", periodic_saves=60,
                     save_pickle_path="data/test.pkl", save_json_path="data/test.json")
    # ry.analyse(stream_utils=su, start_seconds=28)
    # close place: 64, 248
    # scroll: 146, 244
    # scroll up: 394

    ry.continue_analysing(stream_utils=su, start_seconds=28)  # 2
