import pytest
from leetcode.leetcode_combine_solutions.solutions_all_questions import Solutions

data = [
    ([1, 2, 3, 4], [24, 12, 8, 6]),
    ([1, 2, 3, 4], [24, 12, 8, 6]),
    ([1, 2, 3, 4], [24, 12, 8, 6])

]


@pytest.mark.parametrize('nums,output', data)
def test_product_array(nums, output):
    test = Solutions()
    actual_result = test.productArrayExceptSelf(nums)
    assert actual_result == output


subArrayTestInput = [
    ([1, 1, 1], 2, 2),
    ([1, 2, 3], 3, 2)
]


@pytest.mark.parametrize('num,k,output', subArrayTestInput)
def test_sumArraySum(num, k, output):
    test = Solutions()
    actual_result = test.subArrayEqualSum(num, k)
    assert actual_result == output


"""
input_decode_string = [
    ("3[a]2[bc", "aaabcbc"),
    ("3[a2[c]]", "accaccacc")
]

@pytest.skip
@pytest.mark.parametrize('s,output', input_decode_string)
def test_decode_string(s, output):
    test = Solutions()
    actual_result = test.decodeString(s)
    assert actual_result == output
"""