from enum import Enum


class ResizeModeEnum(str, Enum):
    STRETCH = "stretch"
    COVER = "cover"
    CONTAIN = "contain"
    PAD = "pad"
    CROP_CENTER = "crop_center"


class WipeDirectionsEnum(Enum):
    LEFT_TO_RIGHT = "Swipe Left to Right"
    RIGHT_TO_LEFT = "Swipe Right to Left"
    TOP_TO_BOTTOM = "Swipe Top to Bottom"
    BOTTOM_TO_TOP = "Swipe Bottom to Top"


class UpscaleToEnum(Enum):
    LARGER_IMAGE = "Larger Image"
    SMALLER_IMAGE = "Smaller Image"
