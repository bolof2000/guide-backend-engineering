from collections import deque
class LRUCache(object):

    def __init__(self,c):
        self.c = c
        self.mapp = dict()
        self.queue = deque()


    def get(self,key):
        """
        you have the key, you need to get the value
        this key becomes the recently used so should be appended to the front
        if the key doesnt exist, you should return -1
        :param key:
        :return:
        """
        if key in self.mapp:
            val = self.mapp[key]
            self.queue.remove(key)
            self.queue.append(key)

            return val

    def put(self,key,value):

        """
        check if key does not exist in the map
        check if the capacity is full
        delete the least recently used key
        then set the new value for the key
        remove the key from the queue
        append the key to the queue

        if the key exist already
        update its value
        remove the key from the queue
        append the key to the front of the queue
        :param key:
        :param value:
        :return:
        """
        if key not in self.mapp:
            if len(self.queue) == self.c:
                oldest = self.queue.popleft()
                del self.mapp[oldest]

            self.mapp[key] = value
            self.queue.append(key)

        else:
            self.queue.remove(key)
            self.queue.append(key)
            self.mapp[key] = value

