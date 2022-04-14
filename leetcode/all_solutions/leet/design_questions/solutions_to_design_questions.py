"""
design stack
"""


class Stack:

    def __init__(self):
        self.items = []

    def __str__(self):
        return ' '.join([str(i) for i in self.items])

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


class QueueUsingStacks:

    def __init__(self):

        self.queue = []
        self.stack = []

    def push(self,item):
        self.queue.append(item)

    def peek(self):
        if len(self.queue) ==0:
            while self.stack:
                self.queue.append(self.stack.pop())

        return self.queue[-1]

    def pop(self):

        self.peek()
        return self.queue.pop()

    def isEmpty(self):
        if len(self.queue) == 0 and len(self.stack) ==0:
            return True
        return False