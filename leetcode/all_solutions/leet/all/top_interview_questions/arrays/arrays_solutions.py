from typing import List
from collections import deque
from collections import defaultdict
import pdb


class ArraySolutions:

    def plusOne(self, digits: List[int]) -> List[int]:

        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1

                return digits

        return [1] + digits  # in case we have all 9, we need to increase array size by 1 and fill it with 1

    def singleNumber(self, nums: List[int]) -> int:

        mapp = defaultdict(int)
        for num in nums:
            mapp[num] += 1

        for key, val in mapp.items():
            if val == 1:
                return key

    def singleNumberIII(self, nums: List[int]) -> List[int]:

        mapp = defaultdict(int)
        result = []
        for num in nums:
            mapp[num] += 1

        for key, val in mapp.items():
            if val == 1:
                result.append(key)

        return result

    def missingNumber(self, nums: List[int]) -> int:

        n = len(nums)
        summ = n * (n + 1) // 2

        summ2 = 0
        for num in nums:
            summ2 += num

        missingnum = summ - summ2

        return missingnum

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:

        myset = set(nums)
        result = []

        for num in range(1, len(nums) + 1):
            if num not in myset:
                result.append(num)
        return result

    def productExceptSelf(self, nums: List[int]) -> List[int]:

        """
        Revisit this code before interview
        """

        product = [1] * len(nums)
        cumm = 1

        for i in range(len(nums)):
            product[i] = product[i] * cumm
            cumm = nums[i] * cumm

        cumm = 1

        for i in range(len(nums) - 1, -1, -1):
            product[i] = product[i] * cumm
            cumm = cumm * nums[i]

        return product

    def maxProduct(self, nums: List[int]) -> int:

        product = 1
        window_start = 0
        i = 0
        maxlen = 0
        while i < len(nums):
            product *= nums[i]
            maxlen = max(maxlen, product)

    def maxProduct(self, nums: List[int]) -> List[int]:
        result = max(nums)
        currentMax, currentMin = 1, 1

        for n in nums:
            temp = currentMax * n

            currentMax = max(temp, currentMax * n, n)

            currentMin = min(temp, currentMin * n, n)

            result = max(result, currentMax)

        return result

    def houseRobber(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])

        dp = [nums[0], max(nums[0], nums[1])]

        for i in range(2, len(nums)):
            dp.append(max(dp[i - 2] + nums[i], dp[i - 1]))

        return dp.pop()

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:

        currentMax ,currentMin = 1,1
        total = 0
        for n in nums:
            temp = currentMax*n
            if temp < k:
                total +=1
            print(temp)

        return total





test = ArraySolutions()
# print(test.plusOne([1,0]))
# nums = [1, 2, 1, 3, 2, 5]
# print(test.singleNumberIII(nums))
#nums = [2, 3, -2, 4]
#print(test.maxProduct(nums))
nums = [10,5,2,6]
print(test.numSubarrayProductLessThanK(nums,100))
