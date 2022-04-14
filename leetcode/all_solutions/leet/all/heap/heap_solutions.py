"""
Using Min Heap and Max heap

Python heap is default for min heap. to make a min heap behave like a max heap, we multiply the priority element
with negative 1


"""
from heapq import heappop, heappush
import heapq
from collections import defaultdict


def kthLargestElement(nums, k):
    queue = []
    # the priority is the numbers given - min heap is used to remove smaller numbers so the queue is left with
    # bigger numbers

    for num in nums:
        heappush(queue, num)
        if len(queue) > k:
            heappop(queue)
    print(queue)
    return queue[0]


def kthSmallest(nums, k):
    queue = []
    for num in nums:
        heappush(queue, -num)
        if len(queue) > k:
            heappop(queue)
    return -1 * queue[0]


def kClosest(points, k):
    queue = []

    for sub in points:
        x = sub[0]
        y = sub[1]
        distance = -(x * x + y * x)  # negate the priority to make it act like a max heap since default is min heap

        heappush(queue, [distance, sub])

        if len(queue) > k:
            heappop(queue)
    print(queue)
    res = []
    while queue:
        res.append(heappop(queue)[1])

    return res
    """
    the priority is the distance btw x . we need the smallest distance so we use max heap to kick out the big 
    distance 
    """


def topKFrequentElements(nums, k):
    mapp = defaultdict(int)
    queue = []
    for num in nums:
        mapp[num] += 1

    """
    what do i need to kick out?  elements with smaller frequencies- so we use min heap to kick 
    out big freq
    my priority is the frequency 
    """
    for key in mapp.keys():

        heappush(queue, [mapp[key], key])

        if len(queue) > k:
            heappop(queue)
    res = []
    while queue:
        res.append(heappop(queue)[1])
    return res


nums = [3, 2, 1, 5, 6, 4]
print(kthLargestElement([3, 2, 1, 5, 6, 4], 2))
points = [[1, 3], [-2, 2]]
print(kClosest(points, 1))
print(kthSmallest(nums, 2))
print(topKFrequentElements([1, 1, 1, 2, 2, 3], 2))
