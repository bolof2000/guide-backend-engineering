import pytest

@pytest.mark.tag01
def test_remove_duplicates():
    nums = [0,0,1,1,1,2,2]
    assert remove_duplicates(nums) == 3 
    
    