"""
Patterns:
Arrays is sorted
find minimum in a rotated

Approach : Day 1 study
        : Day two solve questions on topic without looking at solutions 

"""
from typing import List


def findFirstElementSmallerThanTarget(nums: List[int], target: int) -> int:
    boundary_index = -1

    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2

        if nums[mid] >= target:
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return boundary_index


def findTheMinimumInRotatedSortedArray(nums: List[int]) -> int:
    boundary_index = -1

    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < nums[-1]:
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return boundary_index


def findPeakOfMountainArray(nums: List[int]):
    left = 0
    right = len(nums) - 1
    boundary_index = -1
    while left <= right:
        mid = (left + right) // 2

        if mid == len(nums) - 1 or nums[mid] >= nums[mid + 1]:
            boundary_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return boundary_index


def searchInRotatedSortedArray(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    boundary_index = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:

            return mid
        else:
            if nums[mid] <= nums[right]:
                if target >= nums[mid] and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target <= nums[mid] and target >= nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1

    return -1


if __name__ == '__main__':
    # nums = [30,40,50,10,20]
    # print(findTheMinimumInRotatedSortedArray(nums))
    nums = [4, 5, 6, 7, 0, 1, 2]
    print(searchInRotatedSortedArray(nums, 0))
