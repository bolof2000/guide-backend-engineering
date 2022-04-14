from leetcode.leetcode_group.leetcode_75 import *

import pytest

def test_two_sum():
    nums= [2,7,11,15]
    assert two_sum(nums,9) == [0,1]

def test_best_time_to_sell_stock():
    nums = [7,1,5,3,6,4]
    assert best_time_to_buy_and_sell_stock(nums) == 5

def test_max_subarray_sum():
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    assert maximum_subarray_sum(nums)== 6