from enum import Enum


class Event(Enum):
    PRED_SELECTED = 1
    START_DRAWING = 2
    END_DRAWING = 3
    PRED_SETTED = 4
    POINT_SETTED = 5
