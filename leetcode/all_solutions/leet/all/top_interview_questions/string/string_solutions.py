from collections import defaultdict
from collections import deque
from typing import List
from heapq import heappop, heappush


class StringSolutions:

    def strStr(self, haystack, needle):

        if len(haystack) == 0 and len(needle) == 0:
            return 0

        for i in range(len(haystack)):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1

    def findTheDifference(self, s: str, t: str) -> str:
        stack = []

        for char in s:
            stack.append(char)
        print(stack)

        for ele in t:
            if ele in stack and len(stack) > 0:
                stack.remove(ele)
            stack.append(ele)

        return stack.pop()

    def compress(self, chars: List[str]) -> int:

        n = len(chars)
        count = 1
        i = 0
        for j in range(1, len(chars)):
            if j < n and chars[i] == chars[j - 1]:
                count += 1
            else:
                chars[i] = chars[j - 1]

                i += 1

            if count > 1:
                for k in str(count):
                    chars[i] = k
                    i += 1
            count = 1

        return i

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        result = []
        print(intervals)
        for start, end in intervals:
            print(start)
            # print(end)
            if not result or start > result[-1][1]:
                print(result)
                result.append([start, end])
            else:
                result[-1][1] = max(result[-1][1], end)

        return result

    def smallestRange(self, nums: List[int], k: int) -> int:
        result = []

        for i in range(len(nums)):
            num = nums[i]

            if num <= k:
                result.append(num + k)
            else:
                result.append(num - k)
        print(result)
        return max(result) - min(result)


test = StringSolutions()
# print(test.strStr("",""))
# print(test.findTheDifference("ae","aea"))
chars = ["a", "a", "b", "b", "c", "c", "c"]
chars2 = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
chars3 = ["a", "a", "a", "b", "b", "a", "a"]
chars4 = ["a"]
# print(test.compress(chars4))
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# print(intervals[-1][0])
# print(intervals)
# print(test.merge(intervals))
nums = [2,7,2]
print(test.smallestRange(nums,1))