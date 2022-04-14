from typing import List


def removeDuplicatesFromSortedLists(nums: List[int]):
    window_start = 0
    size = 0

    for i in range(len(nums)):
        if nums[i] != nums[window_start]:
            size += 1
            window_start = i

    return size + 1


class Node:

    def __init__(self, val, next):
        self.val = val
        self.next = next


def middleOfLinkedList(head: Node):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow.val


def moveZeros(nums: List[int]):
    index = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[index] = nums[i]

            index += 1

    for i in range(index, len(nums)):
        nums[i] = 0


def moveZeros2(nums: List[int]):
    window_start = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[window_start], nums[i] = nums[i], nums[window_start]
            window_start += 1


def longestSubString(string: str):
    distinct = set()
    window_start = 0
    left = 0
    maxlen = 0

    while left < len(string):
        if string[left] not in distinct:
            distinct.add(string[left])
            left += 1
        else:
            maxlen = max(maxlen, left - window_start + 1)
            distinct.remove(string[window_start])
            window_start += 1
    return maxlen


class SmartPhone:

    def __init__(self, manufacturer, model, os, memory, color, price):
        self.manufactuer = manufacturer
        self.model = model
        self.os = os
        self.memory = memory
        self.color = color
        self.price = price

    def show(self):
        final_price = 0
        if self.price > 900:
            discount = self.price * 15 / 100

            final_price = self.price - discount
        return final_price


if __name__ == '__main__':
    iphone = SmartPhone("apple", "iphone13", "IOS", 64, "Black", 1000.00)

    print(iphone.show())
