from enum import Enum


class Event(Enum):
    PRED_COMPUTED = 1
    ONE_SEC_AFTER_DRAWING = 2
    PRED_SELECTED = 3
    START_DRAWING = 4
