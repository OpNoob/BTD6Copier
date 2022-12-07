from enum import Enum


class Event:
    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__


class MouseEvent(Event):
    def __init__(self, x: float, y: float, click=False):
        self.x = x
        self.y = y
        self.click = click

    def getPosition(self):
        return self.x, self.y

    def toDict(self):
        return {
            "type": self.__class__.__name__,
            "x": float(self.x),
            "y": float(self.y),
            "click": self.click,
        }


class ScrollEvent(Event):
    def __init__(self, num_times: int):
        self.num_times = num_times

    def toDict(self):
        return {
            "type": self.__class__.__name__,
            "num_times": self.num_times,
        }


class StartEvent(Event):
    def toDict(self):
        return {
            "type": self.__class__.__name__,
        }


class FastForwardEvent(Event):
    def __init__(self, on: bool = True):
        self.on = on

    def toDict(self):
        return {
            "type": self.__class__.__name__,
            "on": self.on,
        }


class RoundChangeEvent(Event):
    def __init__(self, num):
        self.num = num

    def toDict(self):
        return {
            "type": self.__class__.__name__,
            "num": self.num,
        }


class Events:
    def __init__(self):
        self.data = dict()  # {time_stamp: [events]}

    def addEvent(self, time_stamp, event: Event):
        if event is None:
            return

        if time_stamp in self.data:
            self.data[time_stamp].append(event)
        else:
            self.data[time_stamp] = [event]

    def __iter__(self):
        for time_stamp, events in self.data.items():
            yield time_stamp, events

    def toDict(self):
        return {
            ts: [e.toDict() for e in events] for ts, events in self.data.items()
        }

    def sort(self):
        self.data = dict(sorted(self.data.items()))
        return self.data
