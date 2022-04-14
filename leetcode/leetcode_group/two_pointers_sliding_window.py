from typing import List
from collections import Counter

def remove_duplicates(arr: List[int]) -> int:
    count = 0
    window_start = 0

    for i in range(len(arr)):
        if arr[window_start] != arr[i]:
            count += 1
            window_start = i
        continue
    return count + 1

def move_zeros(nums:List[int])->List[int]: 
    index =0 
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[index] = nums[i]
            index +=1
    
    for i in range(index,len(nums)):
        nums[i] =0 
    return nums
def three_sum(nums:List[int],target:int)->List[List[int]]:
    
    results = []
    nums.sort()
    for i in range(len(nums)-2):
        if i ==0 or (i > 0 and nums[i] != nums[i-1]):
            left = i+1
            right = len(nums)-1
            
            while left < right:
                temp = nums[left]+nums[right]+nums[i]
                if temp == target:
                    results.append([nums[left],nums[right],nums[i]])
                    
                    while left < left and nums[left] == nums[left+1]:
                        left +=1
                    while left < right and nums[right] == nums[right-1]:
                        right -=1

                    left +=1
                    right =-1
                elif temp > target:
                    right -=1
                else:
                    left +=1
    return target
def sub_array_sum(nums:List[int],k:int)->int: 
    dic = Counter()
    dic[0] = 1
    count = 0
    
    current_sum = 0 
    for i in range(len(nums)):
        current_sum += nums[i]
        
        temp = current_sum-k 
        if temp in dic: 
            count += dic[temp]
        dic[current_sum] +=1
    
    return count 
    
                    
    

class Node:
    def __init__(self,val,next=None) -> None:
        self.val = val 
        self.next = next 

def middle_of_linked_list(head:Node)->int: 
    slow = fast = head 
    while fast and fast.next:
        fast = fast.next.next 
        slow = slow.next 
        
    return slow.val 

if __name__ == '__main__':
    nums = [0, 0, 1, 1, 1, 2, 2]
    num2 = [1,0,2,0,0,7]
    #print(remove_duplicates(nums))
    print(move_zeros(num2))
