import copy

from vidgear.gears import CamGear
import cv2
import numpy as np

confidence = 0.9
thresh = 200


def readTemplate(path):
    temp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return temp


def getOutline(img_gray):
    # return img_gray
    img_gray[img_gray > thresh] = 255
    img_gray[img_gray <= thresh] = 0

    # kernel = np.ones((3, 3), np.uint8)
    # outlines = cv2.dilate(img_gray, kernel, iterations=1)

    return img_gray


def connectLines(img_bin, fill=False):
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


mouse_normal_template = readTemplate("assets/mouse 1080p.png")
mouse_normal_mask = connectLines(getOutline(mouse_normal_template.copy()), fill=True)
mouse_click_template = readTemplate("assets/mouse click 1080p.png")
mouse_click_mask = connectLines(getOutline(mouse_click_template.copy()), fill=True)

stream = CamGear(source='https://www.youtube.com/watch?v=R3XUmq8_8j0', stream_mode=True, time_delay=1,
                 logging=True).start()  # YouTube Video URL as input

original_h = 1080
original_w = 1920

stream_h, stream_w, _ = stream.frame.shape
# stream_h, stream_w = (540, 960)


def scale(img):
    if stream_h != original_h or stream_w != original_w:
        resize_h = stream_h / original_h
        resize_w = stream_w / original_w

        h, w = img.shape

        new_h = h * resize_h
        new_w = w * resize_w
        dim = (round(new_w), round(new_h))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)  # cv2.INTER_AREA
        return resized
    else:
        return img


mouse_normal_template = scale(mouse_normal_template)
mouse_normal_mask = scale(mouse_normal_mask)
mouse_click_template = scale(mouse_click_template)
mouse_click_mask = scale(mouse_click_mask)

mouse_normal_template_w, mouse_normal_template_h = mouse_normal_template.shape[::-1]
mouse_click_template_w, mouse_click_template_h = mouse_click_template.shape[::-1]

# cv2.imshow("mouse_normal_template", mouse_normal_template)
# cv2.imshow("mouse_normal_mask", mouse_normal_mask)
# cv2.imshow("mouse_click_template", mouse_click_template)
# cv2.imshow("mouse_click_mask", mouse_click_mask)
# cv2.waitKey(0)
# exit(1)

search_pixels = round(200 * stream_h / original_h)
search_area = None

# infinite loop
while True:
    frame = stream.read()
    if frame is None:
        break

    frame = cv2.resize(frame, (stream_w, stream_h), interpolation=cv2.INTER_CUBIC)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if search_area is None:
        frame_search = frame_gray
    else:
        frame_search = frame_gray[search_area[0][1]:search_area[1][1], search_area[0][0]:search_area[1][0]]
    # frame_outlines = getOutline(frame_gray)

    res_normal = cv2.matchTemplate(frame_search, mouse_normal_template, cv2.TM_CCORR_NORMED, None, mouse_normal_mask)
    normal = np.amax(res_normal)
    res_click = cv2.matchTemplate(frame_search, mouse_click_template, cv2.TM_CCORR_NORMED, None, mouse_click_mask)
    click = np.amax(res_click)

    x_add = 0
    y_add = 0
    if search_area is not None:
        x_add += search_area[0][0]
        y_add += search_area[0][1]

    if normal >= click:
        loc = np.where(res_normal == normal)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, (pt[0] + x_add, pt[1] + y_add), (pt[0] + mouse_normal_template_w + x_add, pt[1] + mouse_normal_template_h + y_add), (0, 0, 255), 2)
            search_area = ((pt[0] - search_pixels + x_add, pt[1] - search_pixels + y_add), (pt[0] + mouse_normal_template_w + search_pixels + x_add, pt[1] + mouse_normal_template_h + search_pixels + y_add))
            cv2.rectangle(frame, search_area[0], search_area[1], (0, 0, 0), 2)
    else:
        loc = np.where(res_click == click)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, (pt[0] + x_add, pt[1] + y_add), (pt[0] + mouse_click_template_w + x_add, pt[1] + mouse_click_template_h + y_add), (255, 0, 0), 2)
            search_area = ((pt[0] - search_pixels + x_add, pt[1] - search_pixels + y_add), (pt[0] + mouse_click_template_w + search_pixels + x_add, pt[1] + mouse_click_template_h + search_pixels + y_add))
            cv2.rectangle(frame, search_area[0], search_area[1], (0, 0, 0), 2)
    # Making sure search area not out of bound
    if search_area is not None:
        if search_area[0][0] < 0:
            search_area = ((0, search_area[0][1]), (search_area[1][0], search_area[1][1]))
        if search_area[0][1] < 0:
            search_area = ((search_area[0][0], 0), (search_area[1][0], search_area[1][1]))
        if search_area[1][0] >= frame_gray.shape[1]:
            search_area = ((search_area[0][0], search_area[0][1]), (frame_gray.shape[1]-1, search_area[1][1]))
        if search_area[1][1] >= frame_gray.shape[0]:
            search_area = ((search_area[0][0], search_area[0][1]), (search_area[1][0], frame_gray.shape[0]-1))


    # cv2.imshow("res", res)
    # cv2.imshow("line", frame_gray)
    cv2.imshow("Output Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
stream.stop()
