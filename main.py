import collections
from collections import *
from typing import List
from heapq import heappush, heappop

import os, sys
utils_path = os.path.join('..', '..', '..', '..', 'problems', 'utils', 'python')
sys.path.append(utils_path)

class LinkedListNode:
  def __init__(self, data):
    self.data = data
    self.next = None
    self.prev = None
    self.arbitrary = None
    
import random

def insert_at_head(head, data):
  newNode = LinkedListNode(data)
  newNode.next = head
  return newNode

def insert_at_tail(head, node):
  if head is None:
    return node

  temp = head;

  while temp.next != None:
    temp = temp.next

  temp.next = node;
  return head

def create_random_list(length):
  list_head = None
  for i in range(0, length):
    list_head = insert_at_head(list_head, random.randrange(1, 100))
  return list_head

def create_linked_list(lst):
  list_head = None
  for x in reversed(lst):
    list_head = insert_at_head(list_head, x)
  return list_head

def display(head):
  temp = head
  while temp:
    print(str(temp.data),end="")
    temp = temp.next
    if temp != None:
      print(", ", end="")

def to_list(head):
  lst = []
  temp = head
  while temp:
    lst.append(temp.data)
    temp = temp.next
  return lst

def is_equal(list1, list2):
  if list1 is list2:
    return True

  while list1 != None and list2 != None:
    if list1.data != list2.data:
      return False
    list1 = list1.next
    list2 = list2.next

  return list1 == list2


class Node():
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.map = {}
        self.deque = deque()

    def get(self, key: int) -> int:
        if key in self.map:
            value = self.map[key]

            self.deque.remove(key)
            self.deque.append(key)

            return value
        else:
            return -1

    def put(self, key: int, value: int):
        if key not in self.map:
            if len(self.deque) == self.capacity:
                oldest = self.deque.popleft()
                del self.map[oldest]
        else:
            self.deque.remove(key)
        self.deque.remove(key)
        self.deque.append(key)


class BinarySearch:

    def binarySearch(self, nums: List[int], target: int):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return -1

    def searchInRotatedSortedArray(self, nums: List[int], target: int):

        left = 0
        right = len(nums) - 1

        while left <= right:

            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            else:

                if nums[mid] <= nums[left]:

                    if nums[left] >= target >= nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1

                else:
                    if nums[mid] <= target <= nums[right]:

                        left = mid + 1
                    else:
                        right = mid - 1

        return -1

    def findMinimumInRotatedSortedArray(self, nums: List[int]):

        left = 0
        right = len(nums) - 1

        boundary_index = -1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] <= nums[-1]:
                boundary_index = mid
                right = right - 1
            else:
                left = mid - 1

        return boundary_index


from math import inf


class TreeNode:

    def __init__(self, left=None, right=None, val=0):
        self.left = left
        self.right = right
        self.val = val


class DepthFirstSearch():
    """
    1) Identify the return Value- Passing value from child to parent
    2) Identify state
    """

    def dfs(self, node: TreeNode, state):
        if node is None:
            return
        left = self.dfs(node.left, state)
        right = self.dfs(node.right, state)

    def maxValueInBinaryTree(self, node: TreeNode):
        def dfs(root: TreeNode, max_val: int):
            if root is None:
                return
            if root.val > max_val:
                max_val = root.val

            dfs(root.left, max_val)
            dfs(root.right, max_val)

        return dfs(node, -inf)

    def depthOfABinaryTree(self, root: TreeNode):
        def dfs(root: TreeNode, depth):
            if root is None:
                return 0

            currentDepth = max(dfs(root.left), dfs(root.right)) + 1
            depth = max(currentDepth, depth)
            return depth

        return dfs(root, -inf)

    def validateBinarySearch(self, root: TreeNode):
        def dfs(node: TreeNode, min_val, max_val):

            if node is None:
                return True
            if not (min_val <= node.val >= max_val):
                return False

            return dfs(node.left, min_val, root.val) and dfs(node.right, root.val, max_val)

        return dfs(root, -inf, inf)


class BacktrackingSolutions:

    def __init__(self):
        pass

    def permute(self, string: str):

        def dfs(path: List[str], used, res):
            if len(path) == len(string):
                res.append("".join(path))
                return
            for i, char in enumerate(string):
                if used[i]:
                    continue
                path.append(char)
                used[i] = True
                dfs(path, used, res)
                path.pop()
                used[i] = False

        res = []
        dfs([], [False] * len(string), res)

        return res

    def letterCombination(self, digits: str):
        KEYBOARD = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }

        def dfs(path: List[str], res: List[str], index):
            if len(path) == len(digits):
                res.append("".join(path))
                return
            key_index = digits[index]
            for char in KEYBOARD[key_index]:
                path.append(char)
                dfs(path, res, index + 1)
                path.pop()

        res = []

        dfs([], res, 0)

        return res

    def combination_sum(self, candidates: List[int], target: int):
        res = []

        def dfs(nums: List[int], remaining, path, index):
            if remaining == 0:
                res.append(path[:])
                return
            for i in range(len(nums)):
                num = nums[i]
                if remaining - num < 0:
                    continue
                dfs(nums, remaining - num, path + [num], i)

        dfs(candidates, target, [], 0)

        return res


class PriorityQuestions:

    def __init__(self):
        self.map = {}
        self.deque = deque()
        self.queue = collections.OrderedDict()

    def KthLargestElement(self, nums: List[int], k: int):

        priority = []

        for num in nums:
            heappush(priority, num)
            if len(priority) > k:
                heappop(priority)
        print(priority)
        return priority[0]

    def topKMostFrequentNumbers(self, nums, param):
        pass

    def topKFrequentWords(self, words: List[str], k: int):

        priority = []

        dic = {}
        for word in words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

        for word in dic.keys():
            heappush(priority, [dic[word], word])
            if len(priority) > k:
                heappop(priority)

        res = []

        for arr in priority:
            res.append(arr[1])

        return res

    def kClosest(self, points, k):
        max_heap = []

        res = []

        for sublist in points:

            x = sublist[0]
            y = sublist[1]

            distance = -(x * x + y * y)

            heappush(max_heap, [distance, sublist])
            if len(max_heap) > k:
                heappop(max_heap)

        for item in max_heap:
            res.append(item[1])

        return res

    def frequencySort(self, string: str):
        s = list(string)
        dic = dict(collections.Counter(s))
        s.sort(key=lambda x: (-dic[x], x))

        return "".join(s)

    def visible_nodes(self, root: TreeNode):
        def dfs(node: TreeNode, max_so_far):
            if not root:
                return 0

            total = 0
            if root.val >= max_so_far:
                total += 1
            total += dfs(node.left, max(root.val, max_so_far))
            total += dfs(node.right, max(root.val, max_so_far))

            return total

        return dfs(root, -inf)

    def validateBFS(self, root: TreeNode):
        def dfs(node: TreeNode, min_val, max_val):
            if not node:
                return True
            if not (min_val <= node.val <= max_val):
                return False

            return dfs(node.left, min_val, root.val) and dfs(node.right, root.val, max_val)

        return dfs(root, -inf, inf)

    def binaryTreetLevelOrderTraversal(self, root: TreeNode):

        res = []
        queue = deque([root])
        while len(queue) > 0:
            n = len(queue)
            path = []
            for _ in range(n):
                node = queue.popleft()
                path.append(node.val)

                for child in [node.left, node.right]:
                    if child is not None:
                        queue.append(child)

            res.append(path)
        return res

    def zigZagLevelOrderTraversal(self, root: TreeNode):
        res = []
        queue = deque([root])
        left_to_right = True

        while len(queue) > 0:

            n = len(queue)
            path = []

            for _ in range(n):
                node = queue.popleft()
                path.append(node.val)

                for child in [node.left, node.right]:
                    if child is not None:
                        queue.append(child)

            if not left_to_right:
                path.reverse()
            res.append(path)

            left_to_right = not left_to_right
        return res

    def inverseTree(self, root: TreeNode):
        def dfs(node: TreeNode):
            if node is None:
                return None
            temp = node.left
            node.left = node.right
            node.right = temp
            dfs(node.left)
            dfs(node.right)

        return dfs(root)

    def reverseLinkedList(self, head: Node):

        prev = None

        current = head

        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp

        return prev

    def inOrderTraversal(self, root: TreeNode):
        res = []

        def dfs(node: TreeNode, res):
            if node is None:
                return
            dfs(node.left, res)
            res.append(node.val)
            dfs(node.left, res)

        return res

    def minAddToMakeValid(self, s: str):
        res = []
        for char in s:
            if char == '(':
                res.append(char)
            elif len(res) > 0 and char == ')' and res[-1] == '(':
                res.pop()
            else:
                res.append(char)
        return len(res)

    def mergeYwoSortedList(self, l1: Node, l2: Node):

        output = Node(0)
        result = output

        while l1 is not None and l2 is not None:

            if l1.val > l2.val:
                output.next = l1
                l1 = l1.next
            else:
                output.next = l2
                l2 = l2.next

            output = output.next

        if l1 is not None:
            output.next = l1
        if l2 is not None:
            output.next = l2

        return result.next

    def addTwoNumbers(self, l1: Node, l2: Node):

        output = Node(0)
        result = output
        carry = 0
        while l1 is not None or l2 is not None:
            if l1 is not None:
                carry += l1.val
                l1 = l1.next
            if l2 is not None:
                carry += l2.val
                l2 = l2.next

            output.next = Node(carry % 10)
            carry = carry / 10

            output = output.next

        return result.next
import collections
from collections import *
from typing import List
from heapq import heappush, heappop


class Node():
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.map = {}
        self.deque = deque()

    def get(self, key: int) -> int:
        if key in self.map:
            value = self.map[key]

            self.deque.remove(key)
            self.deque.append(key)

            return value
        else:
            return -1

    def put(self, key: int, value: int):
        if key not in self.map:
            if len(self.deque) == self.capacity:
                oldest = self.deque.popleft()
                del self.map[oldest]
        else:
            self.deque.remove(key)
        self.deque.remove(key)
        self.deque.append(key)


class BinarySearch:

    def binarySearch(self, nums: List[int], target: int):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return -1

    def searchInRotatedSortedArray(self, nums: List[int], target: int):

        left = 0
        right = len(nums) - 1

        while left <= right:

            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            else:

                if nums[mid] <= nums[left]:

                    if nums[left] >= target >= nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1

                else:
                    if nums[mid] <= target <= nums[right]:

                        left = mid + 1
                    else:
                        right = mid - 1

        return -1

    def findMinimumInRotatedSortedArray(self, nums: List[int]):

        left = 0
        right = len(nums) - 1

        boundary_index = -1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] <= nums[-1]:
                boundary_index = mid
                right = right - 1
            else:
                left = mid - 1

        return boundary_index


from math import inf


class TreeNode:

    def __init__(self, left=None, right=None, val=0):
        self.left = left
        self.right = right
        self.val = val


class DepthFirstSearch():
    """
    1) Identify the return Value- Passing value from child to parent
    2) Identify state
    """

    def dfs(self, node: TreeNode, state):
        if node is None:
            return
        left = self.dfs(node.left, state)
        right = self.dfs(node.right, state)

    def maxValueInBinaryTree(self, node: TreeNode):
        def dfs(root: TreeNode, max_val: int):
            if root is None:
                return
            if root.val > max_val:
                max_val = root.val

            dfs(root.left, max_val)
            dfs(root.right, max_val)

        return dfs(node, -inf)

    def depthOfABinaryTree(self, root: TreeNode):
        def dfs(root: TreeNode, depth):
            if root is None:
                return 0

            currentDepth = max(dfs(root.left), dfs(root.right)) + 1
            depth = max(currentDepth, depth)
            return depth

        return dfs(root, -inf)

    def validateBinarySearch(self, root: TreeNode):
        def dfs(node: TreeNode, min_val, max_val):

            if node is None:
                return True
            if not (min_val <= node.val >= max_val):
                return False

            return dfs(node.left, min_val, root.val) and dfs(node.right, root.val, max_val)

        return dfs(root, -inf, inf)


class BacktrackingSolutions:

    def __init__(self):
        pass

    def permute(self, string: str):

        def dfs(path: List[str], used, res):
            if len(path) == len(string):
                res.append("".join(path))
                return
            for i, char in enumerate(string):
                if used[i]:
                    continue
                path.append(char)
                used[i] = True
                dfs(path, used, res)
                path.pop()
                used[i] = False

        res = []
        dfs([], [False] * len(string), res)

        return res

    def letterCombination(self, digits: str):
        KEYBOARD = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }

        def dfs(path: List[str], res: List[str], index):
            if len(path) == len(digits):
                res.append("".join(path))
                return
            key_index = digits[index]
            for char in KEYBOARD[key_index]:
                path.append(char)
                dfs(path, res, index + 1)
                path.pop()

        res = []

        dfs([], res, 0)

        return res

    def combination_sum(self, candidates: List[int], target: int):
        res = []

        def dfs(nums: List[int], remaining, path, index):
            if remaining == 0:
                res.append(path[:])
                return
            for i in range(len(nums)):
                num = nums[i]
                if remaining - num < 0:
                    continue
                dfs(nums, remaining - num, path + [num], i)

        dfs(candidates, target, [], 0)

        return res


class PriorityQuestions:

    def __init__(self):
        self.map = {}
        self.deque = deque()
        self.queue = collections.OrderedDict()

    def KthLargestElement(self, nums: List[int], k: int):

        priority = []

        for num in nums:
            heappush(priority, num)
            if len(priority) > k:
                heappop(priority)
        print(priority)
        return priority[0]

    def topKMostFrequentNumbers(self, nums, param):
        pass

    def topKFrequentWords(self, words: List[str], k: int):

        priority = []

        dic = {}
        for word in words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

        for word in dic.keys():
            heappush(priority, [dic[word], word])
            if len(priority) > k:
                heappop(priority)

        res = []

        for arr in priority:
            res.append(arr[1])

        return res

    def kClosest(self, points, k):
        max_heap = []

        res = []

        for sublist in points:

            x = sublist[0]
            y = sublist[1]

            distance = -(x * x + y * y)

            heappush(max_heap, [distance, sublist])
            if len(max_heap) > k:
                heappop(max_heap)

        for item in max_heap:
            res.append(item[1])

        return res

    def frequencySort(self, string: str):
        s = list(string)
        dic = dict(collections.Counter(s))
        s.sort(key=lambda x: (-dic[x], x))

        return "".join(s)

    def visible_nodes(self, root: TreeNode):
        def dfs(node: TreeNode, max_so_far):
            if not root:
                return 0

            total = 0
            if root.val >= max_so_far:
                total += 1
            total += dfs(node.left, max(root.val, max_so_far))
            total += dfs(node.right, max(root.val, max_so_far))

            return total

        return dfs(root, -inf)

    def validateBFS(self, root: TreeNode):
        def dfs(node: TreeNode, min_val, max_val):
            if not node:
                return True
            if not (min_val <= node.val <= max_val):
                return False

            return dfs(node.left, min_val, root.val) and dfs(node.right, root.val, max_val)

        return dfs(root, -inf, inf)

    def binaryTreetLevelOrderTraversal(self, root: TreeNode):

        res = []
        queue = deque([root])
        while len(queue) > 0:
            n = len(queue)
            path = []
            for _ in range(n):
                node = queue.popleft()
                path.append(node.val)

                for child in [node.left, node.right]:
                    if child is not None:
                        queue.append(child)

            res.append(path)
        return res

    def zigZagLevelOrderTraversal(self, root: TreeNode):
        res = []
        queue = deque([root])
        left_to_right = True

        while len(queue) > 0:

            n = len(queue)
            path = []

            for _ in range(n):
                node = queue.popleft()
                path.append(node.val)

                for child in [node.left, node.right]:
                    if child is not None:
                        queue.append(child)

            if not left_to_right:
                path.reverse()
            res.append(path)

            left_to_right = not left_to_right
        return res

    def inverseTree(self, root: TreeNode):
        def dfs(node: TreeNode):
            if node is None:
                return None
            temp = node.left
            node.left = node.right
            node.right = temp
            dfs(node.left)
            dfs(node.right)

        return dfs(root)

    def reverseLinkedList(self, head: Node):

        prev = None

        current = head

        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp

        return prev

    def inOrderTraversal(self, root: TreeNode):
        res = []

        def dfs(node: TreeNode, res):
            if node is None:
                return
            dfs(node.left, res)
            res.append(node.val)
            dfs(node.left, res)

        return res

    def minAddToMakeValid(self, s: str):
        res = []
        for char in s:
            if char == '(':
                res.append(char)
            elif len(res) > 0 and char == ')' and res[-1] == '(':
                res.pop()
            else:
                res.append(char)
        return len(res)

    def mergeYwoSortedList(self, l1: Node, l2: Node):

        output = Node(0)
        result = output

        while l1 is not None and l2 is not None:

            if l1.val > l2.val:
                output.next = l1
                l1 = l1.next
            else:
                output.next = l2
                l2 = l2.next

            output = output.next

        if l1 is not None:
            output.next = l1
        if l2 is not None:
            output.next = l2

        return result.next

    def addTwoNumbers(self, l1: Node, l2: Node):

        output = Node(0)
        result = output
        carry = 0
        while l1 is not None or l2 is not None:
            if l1 is not None:
                carry += l1.val
                l1 = l1.next
            if l2 is not None:
                carry += l2.val
                l2 = l2.next

            output.next = Node(carry % 10)
            carry = carry / 10

            output = output.next

        return result.next


from cgitb import reset
from heapq import heappop, heappush
from os import setsid
from typing import List
from functools import lru_cache
import time


def fibo_without_cache(n):
    if n <= 2:
        return n
    return fibo_without_cache(n - 1) + fibo_without_cache(n - 2)


@lru_cache
def fibo_with_cache(n):
    if n <= 2:
        return n
    return fibo_with_cache(n - 1) + fibo_with_cache(n - 2)


def longest_common_subsequence(s1: str, s2: str):
    m, n = len(s1), len(s2)

    @lru_cache(None)
    def dfs(i, j):
        longest = 0
        if i == m and j == n:
            return 0
        if i < m and j < n:
            if s1[i] == s2[j]:
                longest = max(longest, 1 + dfs(i + 1, j + 1))

        if i + 1 < m:
            longest = max(longest, dfs(i + 1, j))
        if j + 1 < n:
            longest = max(longest, dfs(i, j + 1))

        return longest

    return dfs(0, 0)


def word_break(wordDicts, s: str):
    def dfs(index):
        if index == len(s):
            True
        for word in wordDicts:
            if s.startswith(word):
                if dfs(index + len(word)):
                    return True
        return False

    return dfs(0)


from collections import Counter


def top_k_most_frequent_elements(nums, k):
    dic = Counter(nums)
    priority_queue = []
    for item in dic.keys():
        heappush(priority_queue, [dic[item], item])
        if len(priority_queue) > k:
            heappop(priority_queue)

    result = []
    for item in priority_queue:
        result.append(item[1])

    return result


def combination_sum(candidates: List[int], target):
    res = []

    def dfs(start_index, nums, path, remaining):
        if remaining == 0:
            res.append(path.copy())
            return
        for i in range(start_index, len(nums)):
            num = nums[i]
            if remaining - num < 0:
                continue
            dfs(i, nums, path + [num], remaining - num)

    dfs(0, candidates, [], target)
    return res


def combination_sum2(candidates, target):
    res = []

    def dfs(i, path, total):
        if total == target:
            res.append(path.copy())
            return
        if i >= len(candidates) or total > target:
            return

        num = candidates[i]
        path.append(num)
        dfs(i, path, total + num)
        path.pop()
        dfs(i + 1, path, total)

    dfs(0, [], 0)
    return res


def house_robber(nums):
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])

    dp = [nums[0], max(nums[0], nums[1])]

    for i in range(2, len(nums)):
        dp.append(max(dp[i - 2] + nums[i], dp[i - 1]))

    return dp[-1]


def house_robber2(nums):
    return max(nums[0], house_robber(nums[1:]), house_robber(nums[:-1]))


if __name__ == '__main__':
    begin = time.time()
    print(house_robber2([2, 3, 2]))
    # print(fibo_without_cache(30))
    end = time.time()
    print(end - begin)


class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def two_sum(nums: List[int], target: int):
    nums.sort()
    left = 0
    right = len(nums) - 1
    result = []
    while left < right:
        temp = nums[left] + nums[right]
        if temp == target:
            result.append(left)
            result.append(right)
            left += 1
            right -= 1
        elif temp > target:
            right -= 1
        else:
            left += 1
    return result


def best_time_to_buy_and_sell_stock(prices: List[int]):
    profit = 0
    min_price = prices[0]
    for price in prices:
        min_price = min(price, min_price)
        current_profit = price - min_price
        profit = max(profit, current_profit)
    return profit


def contains_duplicate(nums):
    return len(nums) != len(set(nums))


def product_of_array_except_self(nums: List[int]):
    mul = 1
    result = [1] * len(nums)

    for i in range(len(nums) - 1, -1, -1):
        result[i] = mul * result[i]
        mul = mul * nums[i]

    mul = 1

    for i in range(len(nums)):
        result[i] = mul * result[i]
        mul = mul * nums[i]
    return result


def maximum_subarray_sum(nums: List[int]):
    max_sum = 0
    current_sum = nums[0]

    for num in nums:
        current_sum = max(current_sum + num, num)
        max_sum = max(current_sum, max_sum)
    return max_sum


def max_product_subArray():
    pass


def find_minimum_in_rotated_sorted_array(nums: List[int]):
    boundary_index = -1
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] <= nums[-1]:
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return boundary_index


def search_in_rotated_array(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        else:
            if nums[mid] > nums[left]:
                if target < nums[mid] and target > nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if target > nums[mid] and target < nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
    return -1


def missing_number(nums: List[int]):
    num_set = set(nums)
    for i in range(len(nums) + 1):
        if i not in num_set:
            return i


def permute(nums: List[int]):
    res = []

    def dfs(path, used, res):
        if len(path) == len(nums):
            res.append(path)
            return

        for i, num in enumerate(nums):
            if used[i]:
                continue
            path.append(num)
            used[i] = True
            dfs(path, used, res)
            path.pop()
            used[i] = False

    dfs([], [False] * len(nums), res)


from math import inf


def coinChange(coins: List[int], amount: int):
    dp = [inf] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i - coin] + 1, dp[i])
    return dp[-1] if dp[-1] < inf else -1


def add_two_numbers(l1: Node, l2: Node):
    dummy = Node('0')
    result = dummy

    carry = 0
    while l1 or l2 or carry != 0:
        if l1 is not None:
            carry += l1.val
            l1 = l1.next
        if l2 is not None:
            carry += l2.val
            l2 = l2.next
        dummy.next = Node(carry % 10)
        carry = carry / 10

        dummy = dummy.next

    return result.next


def word_break(wordDict: List[str], s: str):
    def dfs(index):
        if index == len(s):
            return True
        for word in wordDict:
            if s[index:].startswith(word):
                if dfs(index + len(word)):
                    return True

        return False

    return dfs(0)


def generate_parenthesis(n: int):
    res = []

    def dfs(path, open, close):
        if open == close == n:
            res.append("".join(path))
            return
        if open < n:
            path.append("(")
            dfs(path, open + 1, close)
            path.pop()
        if close < open:
            path.append(")")
            dfs(path, open, close + 1)
            path.pop()

    dfs([], 0, 0)
    return res


def number_of_islands(grid):
    def dfs(grid, row, col):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == "0":
            return

        grid[row][col] = "0"
        dfs(grid, row + 1, col)
        dfs(grid, row - 1, col)
        dfs(grid, row, col + 1)
        dfs(grid, row, col - 1)

    count = 0

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1":
                count += 1
                dfs(grid, row, col)

    return count


def max_product_subarray(nums):
    res = max(nums)
    current_min, current_max = 1, 1
    for num in nums:
        if num == 0:
            current_max, current_min = 1, 1

        temp = num * current_max

        current_max = (num * current_max, num * current_min, num)
        current_min = (temp, num * current_min, num)

        res = max(current_max, res)
    return res


def find_min_in_rotated_sorted_array(nums: List[int]):
    boundary_index = -1
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] <= nums[-1]:  # confirm if array is still sorted
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return boundary_index


def missing_numbers(nums: List[int]):
    myset = set(nums)

    for i in range(len(nums) + 1):
        if i not in myset:
            return i


def anagaram(s1: str, s2: str):
    priority_queue = []

    for char in s1:
        heappush(priority_queue, char)

    for i in range(len(s2)):
        char = s2[i]
        while char in priority_queue:
            heappop(priority_queue)

    # print(priority_queue)
    if len(priority_queue) == 0:
        return True
    else:
        return False


def climbing_stairs(n):
    if n <= 3:
        return n
    dp = [0, 1, 2]

    for i in range(3, n + 1):
        dp.append(dp[i - 1] + dp[i - 2])

    return dp[-1]


def coin_change(coins, amounts):
    dp = [inf] * (amounts + 1)
    dp[0] = 0

    for amount in range(1, amounts + 1):
        for coin in coins:
            if amount - coin >= 0:
                dp[amount] = min(dp[amount - coin] + 1, dp[amount])

    return dp[-1] if dp[-1] < inf else -1


def longest_increasing_subsequence(nums):
    dp = [1] * len(nums)
    best = 0
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] >= nums[i]:
                continue
            dp[i] = max(dp[i], 1 + dp[j])
        best = max(best, dp[i])
    return best


def unique_path(m:int,n:int):
    dp = [[1 for row in range(n)] for col in range(m)]

    for row in range(1,m):
        for col in range(1,n):
            dp[row][col] = dp[row][col-1]+dp[row-1][col]

    print(dp)
    return dp[-1][-1]


def jump_game(nums):
    dp = [False]*len(nums)
    pass 

def letter_com_phone(digits):
    
    KEYBOARD = {
    '2': 'abc',
    '3': 'def',
    '4': 'ghi',
    '5': 'jkl',
    '6': 'mno',
    '7': 'pqrs',
    '8': 'tuv',
    '9': 'wxyz'
    }
    res = []
    
    def dfs(path,index):
        if len(path) == len(digits):
            res.append("".join(path))
            return 
        
        for char in KEYBOARD[digits[index]]:
            path.append(char)
            dfs(path,index+1)
            path.pop()
    dfs([],0)
    return res 

def jump_game(nums):
    goal = len(nums)-1
    
    for i in range(len(nums)-1,-1,-1):
        if i+nums[i] >= goal:
            goal = i 
            
    return True if goal==0 else  False 

        
def can_attend_meetings(intervals):
    intervals.sort(key=lambda x:x[0])
    
    pass

def merge_interval(intervals):
    intervals.sort(key=lambda x:x[0])
    
    result = [intervals[0]]
    
    for sublist in intervals:
        last = result[-1]
        if sublist[0] <= last[1]:
            last[1] = max(sublist[1],last[1])
        else:
            result.append(sublist)

    return result 

class QueueUsingStack:
    
    def __init__(self) -> None:
        self.stack1 = []
        self.stack2 = []
        
    def push(self,x):
        self.stack1.append(x)
    def pop(self):
        self.peek()
        return self.stack2.pop()
        
    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
    
        return self.stack2[-1]
                
    def empty(self):
        return True if len(self.stack1)==0 and len(self.stack2)==0 else False 
    

def meeting_room_two(intervals):
   start = []
   end = []
   count = 0
   
   for item in intervals:
       start = item[0]
       end = item[1]
       
   start.sort()
   end.sort()
   
   j = 0
   for i in range(len(start)):
       
       if start[i] < end[j]:
           count +=1
       else:
           j +=1

   return count 
           
def meeting_room_lam(intervals:List[List[int]]):
    intervals.sort(key=lambda x:(x[0],x[1]))
    
    return intervals


def group_titles(strs):
    res = {}
    count = [0]*26 
    for s in strs:
       
        for c in s:
            index = ord(c) - ord('a')
            count[index] += 1

        key = tuple(count)
        print(key)

    return count 
           

def merge2_linkedlist(l1,l2):
    dummy = LinkedListNode(-1)
    result = dummy 
    while l1 and l2:
        if l1.data < l2.data:
            dummy.next = l1
            l1 = l1.next
        else:
            dummy.next = l2
            l2 = l2.next 

        dummy = dummy.next 
    
    if l1 is not None: 
        dummy.next = l1 
    else:
        dummy.next = l2
    return result.next 

def mergeK_list(lists):
    if len(lists) >0:
        res = lists[0]
        for i in range(1,len(lists)):
            res = merge2_linkedlist(res,lists[i])
        return res 
    return

def find_minimum_in_rotated_sorted_array(nums):
    
    left,right = 0, len(nums)-1
    if nums(left) < nums[right]:
        return nums[left]

    while left+1 < right:
        mid = (left+right)//2
        if nums[mid] < nums[right]:
            right =mid 
        else:
            left = mid 
    return min(nums[left],nums[right])


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers)-1
        result = []
        while left< right:
            temp= numbers[left]+numbers[right]
            if temp == target:
                result.append(left+1)
                result.append(right+1)

                left +=1
                right -=1
            elif temp > target:
                right -=1
            else:
                left +=1
        return result 
    
    
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        index =0 
        for i in range(len(nums)):
            if nums[i] !=0:
                nums[index] = nums[i]
                index +=1
        for i in range(index,len(nums)):
            nums[i] =0 
 
     
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        
        window = 0 
        for i in range(1,len(nums)):
            if nums[window] != nums[i]:
                windows +=1
                nums[window] = nums[i] 
        return window 
             

    
if __name__ == '__main__':

    #print(unique_path(3,7))
    intervals = [[0,30],[5,10],[15,20]]
    #print(meeting_room_lam(intervals))
    
    #titles = ["duel","dule","speed","spede","deul","cars"]
   # print(group_titles(titles))
    a = create_linked_list([11,41,51])
    b = create_linked_list([21,23,42])
    c = create_linked_list([25,56,66,72])
    d = create_linked_list([50,60,70])
    display(mergeK_list([a,b,c,d]))
    





