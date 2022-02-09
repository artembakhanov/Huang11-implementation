from message import Message, MessageType


class Network:
    def __init__(self, verbose=False):
        self.all_stats = 0

        self.stat = dict(zip(MessageType, [0] * len(MessageType)))
        self.verbose = verbose

    def send_message(self, message: Message, from_: 'Node', to: 'Node'):
        self.all_stats += len(message)
        self.stat[message.type] += len(message)
        to.recv_message(message, from_)

        if self.verbose:
            print(f"A {message.type} message has been sent from {from_} to {to}")

    def reset(self):
        self.all_stats = 0

        for k in self.stat.keys():
            self.stat[k] = 0
