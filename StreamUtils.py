import static
from Utils import Utils

from vidgear.gears import CamGear
import cv2
import numpy as np


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
        scaled_area = ((self.width * area[0][0]), self.height * area[0][1]), (self.width * area[1][0], self.height * area[1][1])
        if integer:
            scaled_area = ((round(scaled_area[0][0]), round(scaled_area[0][1])), (round(scaled_area[1][0]), round(scaled_area[1][1])))
        return scaled_area

    def _getImageSearch(self, image, norm_area: tuple[tuple, tuple]):
        area = self._scaleStatic(norm_area)
        image_search = image[area[0][1]: area[1][1], area[0][0]:area[1][0]]
        return image_search

    def check_play(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]), self._play_template)

    def check_victory(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["victory"]), self._victory_template)

    def check_fast_forward(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]), self._fast_forward_template)

    def check_no_fast_forward(self, image_gray):
        return self._find(self._getImageSearch(image_gray, static.search_locations["play"]), self._no_fast_forward_template)

    def readYoutube(self, url, show_frames=True, rescale_out_frames=(960, 540)):
        # Start stream to extract height and width
        stream = CamGear(source=url, stream_mode=True, time_delay=1, logging=True).start()
        self.height, self.width, _ = stream.frame.shape

        # Scaling
        mouse_normal_template = self.scale(self._mouse_normal_template.copy())
        mouse_normal_mask = self.scale(self._mouse_normal_mask.copy())
        mouse_click_template = self.scale(self._mouse_click_template.copy())
        mouse_click_mask = self.scale(self._mouse_click_mask.copy())

        # Get shapes of templates
        mouse_normal_template_w, mouse_normal_template_h = mouse_normal_template.shape[::-1]
        mouse_click_template_w, mouse_click_template_h = mouse_click_template.shape[::-1]

        # Searching pixels for mouse (So no need to search in all the image)
        search_pixels = round(self.search_mouse_pixels * self.height / static.base_resolution_height)
        search_area = None

        # infinite loop
        while True:
            frame = stream.read()
            if frame is None:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.check_victory(frame_gray):
                break

            play_location = self._scaleStatic(static.search_locations["play"])
            if self.check_fast_forward(frame_gray):
                cv2.rectangle(frame, play_location[0], play_location[1], (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, play_location[0], play_location[1], (0, 0, 255), 2)

            if search_area is None:
                frame_search = frame_gray
            else:
                frame_search = frame_gray[search_area[0][1]:search_area[1][1], search_area[0][0]:search_area[1][0]]
            # frame_outlines = getOutline(frame_gray)

            res_normal = cv2.matchTemplate(frame_search, mouse_normal_template, cv2.TM_CCORR_NORMED, None,
                                           mouse_normal_mask)
            normal = np.amax(res_normal)
            res_click = cv2.matchTemplate(frame_search, mouse_click_template, cv2.TM_CCORR_NORMED, None,
                                          mouse_click_mask)
            click = np.amax(res_click)

            x_add = 0
            y_add = 0
            if search_area is not None:
                x_add += search_area[0][0]
                y_add += search_area[0][1]

            if normal >= click:
                loc = np.where(res_normal == normal)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, (pt[0] + x_add, pt[1] + y_add),
                                  (pt[0] + mouse_normal_template_w + x_add, pt[1] + mouse_normal_template_h + y_add),
                                  (0, 0, 255), 2)
                    search_area = ((pt[0] - search_pixels + x_add, pt[1] - search_pixels + y_add), (
                        pt[0] + mouse_normal_template_w + search_pixels + x_add,
                        pt[1] + mouse_normal_template_h + search_pixels + y_add))
                    cv2.rectangle(frame, search_area[0], search_area[1], (0, 0, 0), 2)
            else:
                loc = np.where(res_click == click)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, (pt[0] + x_add, pt[1] + y_add),
                                  (pt[0] + mouse_click_template_w + x_add, pt[1] + mouse_click_template_h + y_add),
                                  (255, 0, 0), 2)
                    search_area = ((pt[0] - search_pixels + x_add, pt[1] - search_pixels + y_add), (
                        pt[0] + mouse_click_template_w + search_pixels + x_add,
                        pt[1] + mouse_click_template_h + search_pixels + y_add))
                    cv2.rectangle(frame, search_area[0], search_area[1], (0, 0, 0), 2)
            # Making sure search area not out of bound
            if search_area is not None:
                if search_area[0][0] < 0:
                    search_area = ((0, search_area[0][1]), (search_area[1][0], search_area[1][1]))
                if search_area[0][1] < 0:
                    search_area = ((search_area[0][0], 0), (search_area[1][0], search_area[1][1]))
                if search_area[1][0] >= frame_gray.shape[1]:
                    search_area = ((search_area[0][0], search_area[0][1]), (frame_gray.shape[1] - 1, search_area[1][1]))
                if search_area[1][1] >= frame_gray.shape[0]:
                    search_area = ((search_area[0][0], search_area[0][1]), (search_area[1][0], frame_gray.shape[0] - 1))

            # cv2.imshow("res", res)
            # cv2.imshow("line", frame_gray)
            if show_frames:
                if rescale_out_frames is not None:
                    frame = cv2.resize(frame, rescale_out_frames, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Output Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break


if __name__ == "__main__":
    su = StreamUtils()
    su.readYoutube("https://www.youtube.com/watch?v=R3XUmq8_8j0")
