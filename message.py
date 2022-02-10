import enum
from abc import abstractmethod, ABC

from result import Result_, Result


class MessageType(enum.IntEnum):
    NEW_NODE = -1
    NEW_TASK = 0
    RESULT = 1

    @classmethod
    def all(cls):
        return list(map(int, cls))


class Message(ABC):
    data = None

    @property
    @abstractmethod
    def type(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class NewTaskMessage(Message):
    type = MessageType.NEW_TASK

    def __init__(self, eps: float):
        self.data = eps

    def __len__(self):
        return 1


class ResultMessage(Message):
    type = MessageType.RESULT

    def __init__(self, result: Result):
        self.data = result

    def __len__(self):
        return len(self.data)


class NewNodeMessage(Message):
    type = MessageType.NEW_NODE

    def __init__(self, data_length):
        self.data = data_length

    def __len__(self):
        return 1
