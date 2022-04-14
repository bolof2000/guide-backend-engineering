class BinarySearchImplementation(object):

    def search(self, arr, target):
        """
        Search in rotated sorted array .
        check the mid points and compare with the nums[left] and nums[right]

        """
        right = len(arr)-1
        left =0
        while left <= right :

            mid = (left+right)//2

            if arr[mid] == target:
                return mid

            else:

                if arr[mid] >= arr[left]:
                    if arr[mid] >= target >= arr[left]:
                        right = mid-1
                    else:
                        left = mid+1

                else:

                    if arr[mid] <= target <= arr[right]:
                        left = mid+1
                    else:
                        right = mid-1

        return -1








    def findMin(self, nums):


        """
         Given the sorted rotated array nums of unique elements, return the minimum el
# ement of this array.
#
#
#  Example 1:
#
#
# Input: nums = [3,4,5,1,2]
# Output: 1
# Explanation: The original array was [1,2,3,4,5] rotated 3 times.
        :param nums:
        :return:
        """

        left = 0
        right = len(nums) - 1
        boundary_index = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= nums[-1]:
                boundary_index = mid
                right = mid - 1
            else:
                left = mid + 1

        return nums[boundary_index]

    def peakIndexInMountainArray(self, arr):
        pass

    def searchRange(self, nums, target):
        """
            define two methods to search when target is greater than mid val and target is
            less than mid value. move left and right pointers accordingly

        """

        result = [self.find_right_index(nums,target),self.find_left_index(nums,target)]
        return sorted(result)

    def find_right_index(self, nums, target):

        index = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                index = mid

            if target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return index

    def find_left_index(self, nums, target):

        index = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                index = mid

            if target >= nums[mid]:
                left = mid + 1
            else:
                right = mid - 1

        return index

    def first_not_smaller(self, nums, target):

        """
            find midpoint
            set boundary to -1
            compare number at midpoint to the target
            if true, set boundary to mid
            move pointers accordingly

        """

        right = len(nums) - 1
        left = 0
        boundary_index = -1

        while left <= right:
            mid = (left + right) // 2
            if target <= nums[mid]:
                boundary_index = mid
                right = mid - 1

            else:
                left = mid + 1

        return boundary_index

    def find_first_occurence(self, nums, target):

        """
      Given a sorted array of integers and a target integer, find the first occurrence of the target and return its index. Return -1 if the target is not in the array.
Input:
arr = [1, 3, 3, 3, 3, 6, 10, 10, 10, 100]
target = 3
Output:

1
"""
        left = 0
        right = len(nums) - 1
        boundary_index = -1
        while left <= right:
            mid = (left + right) // 2

            if target == nums[mid]:
                boundary_index = mid
                right = mid - 1

            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return boundary_index




test_search = BinarySearchImplementation()
arr = [5,7,7,8,8,10]
print(test_search.search(arr,8))
print(test_search.find_left_index(arr,8))
print(test_search.find_right_index(arr,8))

