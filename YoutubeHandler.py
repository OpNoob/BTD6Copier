# import cap_from_youtube
import yt_dlp


class VideoStream:
    url: str = None
    resolution: str = None
    height: int = 0
    width: int = 0

    def __init__(self, video_format):
        self.url = video_format['url']
        self.resolution = video_format['format_note']
        self.height = video_format['height']
        self.width = video_format['width']

    def __str__(self):
        return f'{self.resolution} ({self.height}x{self.width}): {self.url}'

    def __repr__(self):
        return str(self)

    def isResolution(self, resolution: str | tuple[int, int]) -> bool:
        if isinstance(resolution, str):
            return self._isResolutionS(resolution)
        elif isinstance(resolution, tuple):
            return self._isResolutionHW(resolution)

    def _isResolutionHW(self, resolution: tuple[int, int]) -> bool:
        """
        Matches resolution check
        :param resolution: (height, width)
        :return: bool
        """
        return resolution[0] == self.height and resolution[1] == self.width

    def _isResolutionS(self, resolution: str) -> bool:
        """
        Matches resolution check
        :param resolution: string
        :return: bool
        """
        return resolution == self.resolution


class YtInfo:
    def __init__(self, url, download=False):
        ydl_opts = {}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            self._info = ydl.extract_info(url, download=download)

            self.title = self._info["title"]

            self.video_streams = [VideoStream(format) for format in self._info['formats'][::-1] if
                                  format['vcodec'] != 'none']

    def getStream(self, resolution):
        for x in self.video_streams:
            if x.isResolution(resolution):
                return x

    def __repr__(self):
        return str(self._info)


if __name__ == "__main__":
    yi = YtInfo("https://www.youtube.com/watch?v=iAPWkferxe8")
    print(yi.getStream("720p"))
    # print(yi)
    # print(yi.title)
