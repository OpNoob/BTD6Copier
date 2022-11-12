from enum import Enum


class EventType(Enum):
    start = 0
    mouse = 1
    scroll = 2
    fast_forward = 3


class Event:
    def __init__(self, typ):
        self.type = typ

    def toDict(self):
        return {
            "type": self.type.name
        }

    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__


class MouseEvent(Event):
    def __init__(self, x: float, y: float, click=False):
        self.x = x
        self.y = y
        self.click = click
        Event.__init__(self, EventType.mouse)

    def getPosition(self):
        return self.x, self.y

    def toDict(self):
        return {
            "type": self.type.name,
            "x": float(self.x),
            "y": float(self.y),
            "click": self.click,
        }


class ScrollEvent(Event):
    def __init__(self, num_times: int):
        Event.__init__(self, EventType.scroll)
        self.num_times = num_times

    def toDict(self):
        return {
            "type": self.type.name,
            "num_times": self.num_times,
        }


class StartEvent(Event):
    def __init__(self):
        Event.__init__(self, EventType.start)

    def toDict(self):
        return {
            "type": self.type.name,
        }


class FastForwardEvent(Event):
    def __init__(self, on: bool = True):
        self.on = on
        Event.__init__(self, EventType.fast_forward)

    def toDict(self):
        return {
            "type": self.type.name,
            "on": self.on,
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
