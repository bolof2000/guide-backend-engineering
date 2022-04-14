import pytest
from leetcode.leetcode_group import heap_queue
from leetcode.leetcode_group.heap_queue import *
from leetcode.leetcode_group.two_pointers_sliding_window import *

@pytest.mark.tag01
def test_remove_duplicates():
    nums = [0, 0, 1, 1, 1, 2, 2]
    assert remove_duplicates(nums) == 3


@pytest.mark.tag02
def test_move_zero():
    num2 = [1, 0, 2, 0, 0, 7]
    assert move_zeros(num2) == [1, 2, 7, 0, 0, 0]


@pytest.mark.tag03
def test_sub_array_sum():
    nums = [1, 1, 1]
    assert sub_array_sum(nums, 2) == 2


@pytest.mark.tag04
def test_k_closest_point():
    points = [[3, 3], [5, -1], [-2, 4]]
    assert K_closest_points(points, 2) == [[-2, 4], [3, 3]]


@pytest.mark.tag05
def test_kth_largest_element_in_array():
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    assert kth_largest_element_in_array(nums, k) == 5


def test_top_kth_words():
    keys = [1, 1, 1, 2, 2, 3]
    assert sorted(top_k_frequent_elements(keys, 2)) == [1, 2]


weekdays1 = ['mon', 'tue', 'wed']
weekdays2 = ['fri', 'sat', 'sun']


@pytest.fixture()
def setup_data():
    weekdays1 = ['mon', 'tue', 'wed']
    weekdays2 = ['fri', 'sat', 'sun']
    weekdays1.append('thur')
    yield weekdays1
    print("\n After yield in setup fixture")
    weekdays1.pop()  # clean up after data is used 


@pytest.mark.tag06
def test_extendList(setup_data):
    setup_data.extend(weekdays2)
    assert setup_data == ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
