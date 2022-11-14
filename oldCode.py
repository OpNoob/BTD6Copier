def motionTrack(self, image_gray_previous, image_gray, image_write=None):
    h, w = image_gray.shape

    # Initiate ORB detector
    orb = cv2.ORB_create()  # edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image_gray_previous, None)
    kp2, des2 = orb.detectAndCompute(image_gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(image_gray_previous, kp1, image_gray, kp2, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    x_sum_diff = 0
    y_sum_diff = 0
    for m in matches:
        x_sum_diff += kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
        y_sum_diff += kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]
    x_sum_diff /= len(matches)
    y_sum_diff /= len(matches)

    single_scroll_pixel_shift = static.pixels_scroll * self.height

    if y_sum_diff > single_scroll_pixel_shift / 2:
        self._scroll_pixel_total += y_sum_diff

    num_scrolls = math.ceil(self._scroll_pixel_total / single_scroll_pixel_shift)
    print("num_scrolls: ", num_scrolls)
    print(self._scroll_pixel_total / single_scroll_pixel_shift)

    # scroll_ratio = y_sum_diff / h
    # print(scroll_ratio)

    cv2.imshow("test", img3)


    def readYoutube(self, youtube_url, show_frames=True, rescale_out_frames=(960, 540)):
        resolution = "1080p60"

        # Start stream to extract height and width
        streams, resolutions = cap_from_youtube.list_video_streams(youtube_url)
        if resolution in resolutions:
            res_index = np.where(resolutions == resolution)[0][0]
            stream_url = streams[res_index].url
            cap = cv2.VideoCapture(stream_url)
        else:
            raise ValueError(f'Resolution {resolution} not available')

        # Skip to desired testing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 8700)

        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

        # Scrolling
        points_previous = None
        scroll_previous = 0

        # Check if started
        started = True

        fps = cap.get(cv2.CAP_PROP_FPS)

        # infinite loop
        while True:
            ret, frame = cap.read()
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            time_stamp = frame_no / fps  # In seconds

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not started and self.check_play(frame_gray):
                started = True

            if started:
                # Victory check
                if self.check_victory(frame_gray):
                    break

                # Fast-forward check
                play_location = self._scaleStatic(static.search_locations["play"])
                if self.check_fast_forward(frame_gray):
                    cv2.rectangle(frame, play_location[0], play_location[1], (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, play_location[0], play_location[1], (0, 0, 255), 2)

                # Calculate scroll
                points = self._find_scroll(frame_gray, image_write=frame)
                if points_previous is not None:
                    y_shift = 0
                    match_count = 0
                    for index, (found_x, found_y, _, _) in points.items():
                        if index in points_previous:
                            (previous_x, previous_y, _, _) = points_previous[index]
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
                    scroll_add = num_scrolls - scroll_previous
                    scroll_previous = num_scrolls
                    # print("scroll_add: ", scroll_add)
                points_previous = points

                # Mouse clicks
                if search_area is None:
                    frame_search = frame_gray
                else:
                    frame_search = frame_gray[search_area[0][1]:search_area[1][1], search_area[0][0]:search_area[1][0]]
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
                                      (
                                          pt[0] + mouse_normal_template_w + x_add,
                                          pt[1] + mouse_normal_template_h + y_add),
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
                        search_area = (
                            (search_area[0][0], search_area[0][1]), (frame_gray.shape[1] - 1, search_area[1][1]))
                    if search_area[1][1] >= frame_gray.shape[0]:
                        search_area = (
                            (search_area[0][0], search_area[0][1]), (search_area[1][0], frame_gray.shape[0] - 1))

            if show_frames:
                if rescale_out_frames is not None:
                    frame = cv2.resize(frame, rescale_out_frames, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Output Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()

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