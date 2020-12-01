import enum

class ProblemTypes(enum.Enum):
    ContinuousControl = enum.auto()
    Classification = enum.auto()
    Regression = enum.auto()
    Undefined = enum.auto()
