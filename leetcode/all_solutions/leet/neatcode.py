"""
Neatcode solutions - Updated daily

Binary Search
DFS
Strings
Arrays
Backtracking
Two Pointers
Queue/Heap
DP
Divide and Conquer
Systems Design

----
Graph
BFS

"""
from typing import List
import pdb
def subArraySumsEqualK(nums:List[int],k:int):

    res = 0
    curSum =0
    dic = {0:1}

    for num in nums:
        curSum +=num
        diff = curSum-k
        #pdb.set_trace()
        res += dic.get(diff,0)

        dic[curSum] += dic.get(curSum,0)

    return res

def groupAnagrams(strs:List[str]):

    dic = {}
    res = []

def hashed(string:str):
    return "".join(string[::-1])
