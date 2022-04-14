from cgitb import reset
from heapq import heappop, heappush
from os import setsid
from typing import List
from functools import lru_cache
import time 

def fibo_without_cache(n):
    if n <=2:
        return n
    return fibo_without_cache(n-1)+fibo_without_cache(n-2)

@lru_cache
def fibo_with_cache(n):
    if n <=2:
        return n 
    return fibo_with_cache(n-1)+fibo_with_cache(n-2) 


def longest_common_subsequence(s1:str,s2:str):
    m,n = len(s1),len(s2)
    
    @lru_cache(None)
    def dfs(i,j):
        longest = 0
        if i== m and j == n:
            return 0 
        if i < m and j < n: 
            if s1[i] == s2[j]:
                longest = max(longest,1+dfs(i+1,j+1))

        if i+1 < m: 
            longest = max(longest,dfs(i+1,j))
        if j+1 < n: 
            longest = max(longest,dfs(i,j+1))   
        
        return longest 
    return dfs(0,0)         
  
def word_break(wordDicts,s:str):
    
    def dfs(index):
        if index == len(s):
            True 
        for word in wordDicts:
            if s.startswith(word):
                if dfs(index+len(word)):
                    return True 
        return False 
    
    return dfs(0)
  
from collections import Counter 
def top_k_most_frequent_elements(nums,k):
    dic = Counter(nums)
    priority_queue = []
    for item in dic.keys():
        heappush(priority_queue,[dic[item],item])
        if len(priority_queue) > k:
            heappop(priority_queue)
            
    result = []
    for item in priority_queue:
        result.append(item[1])

    return result

def combination_sum(candidates:List[int],target):
    res = []
    def dfs(start_index,nums,path,remaining):
        if remaining ==0:
            res.append(path.copy())
            return 
        for i in range(start_index,len(nums)):
            num = nums[i]
            if remaining-num < 0:
                continue
            dfs(i,nums,path+[num],remaining-num)
    
    dfs(0,candidates,[],target)
    return res 
def combination_sum2(candidates,target):
    res = []
    def dfs(i,path,total):
        if total == target:
            res.append(path.copy())
            return 
        if i >= len(candidates) or total > target:
            return 
        
        num = candidates[i]
        path.append(num)
        dfs(i,path,total+num)
        path.pop()
        dfs(i+1,path,total)
        
    dfs(0,[],0)
    return res 
 
def house_robber(nums):
     if len(nums) == 1:
         return nums[0]
     if len(nums) ==2:
         return max(nums[0],nums[1])
     
     dp = [nums[0],max(nums[0],nums[1])]
     
     for i in range(2,len(nums)):
         dp.append(max(dp[i-2]+nums[i],dp[i-1]))
         
     return dp[-1]   

def house_robber2(nums):
    return max(nums[0],house_robber(nums[1:]),house_robber(nums[:-1]))
     
      
if __name__ == '__main__':
    begin = time.time()
    print(house_robber2([2,3,2]))
    #print(fibo_without_cache(30))
    end = time.time()
    print(end-begin)
     
    

class Node:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next

def two_sum(nums:List[int],target:int):
    nums.sort()
    left = 0
    right = len(nums)-1
    result = []
    while left < right:
        temp = nums[left] + nums[right]
        if temp == target:
            result.append(left)
            result.append(right)
            left +=1
            right -=1
        elif temp > target:
            right -=1
        else:
            left +=1
    return  result

def best_time_to_buy_and_sell_stock(prices:List[int]):
    profit =0
    min_price =prices[0]
    for price in prices:
        min_price = min(price,min_price)
        current_profit = price-min_price
        profit = max(profit,current_profit)
    return profit

def contains_duplicate(nums):
    return len(nums) != len(set(nums))

def product_of_array_except_self(nums:List[int]):
    mul = 1
    result = [1]*len(nums)

    for i in range(len(nums)-1,-1,-1):
        result[i] = mul * result[i]
        mul = mul*nums[i]

    mul = 1

    for i in range(len(nums)):
        result[i] = mul * result[i]
        mul = mul * nums[i]
    return result

def maximum_subarray_sum(nums:List[int]):
    max_sum = 0
    current_sum = nums[0]

    for num in nums:
        current_sum = max(current_sum+num,num)
        max_sum = max(current_sum,max_sum)
    return max_sum
def max_product_subArray():
    pass

def find_minimum_in_rotated_sorted_array(nums:List[int]):
    boundary_index = -1
    left = 0
    right = len(nums)-1
    while left <= right:
        mid = (left+right)//2
        if nums[mid] <= nums[-1]:
            boundary_index = mid
            right= mid-1
        else:
            left = mid+1
    return boundary_index

def search_in_rotated_array(nums,target):
    left =0
    right = len(nums)-1
    while left <= right:
        mid = (left+right)//2

        if nums[mid] == target:
            return mid
        else:
            if nums[mid] > nums[left]:
                if target < nums[mid] and target > nums[left]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if target > nums[mid] and target< nums[right]:
                    left = mid+1
                else:
                    right = mid-1
    return -1

def missing_number(nums:List[int]):
    num_set = set(nums)
    for i in range(len(nums)+1):
        if i not in num_set:
            return i

def permute(nums:List[int]):
    res = []
    def dfs(path,used,res):
        if len(path) == len(nums):
            res.append(path)
            return

        for i,num in enumerate(nums):
            if used[i]:
                continue
            path.append(num)
            used[i] = True
            dfs(path,used,res)
            path.pop()
            used[i] = False
    dfs([],[False]*len(nums),res)

from math import inf
def coinChange(coins:List[int],amount:int):
    dp = [inf]* (amount+1)
    dp[0] = 0
    for i in range(1,amount+1):
        for coin in coins:
            if i-coin >=0:
                dp[i] = min(dp[i-coin]+1,dp[i])
    return dp[-1] if dp[-1] < inf else  -1

def add_two_numbers(l1:Node,l2:Node):
    dummy = Node('0')
    result = dummy

    carry = 0
    while l1 or l2 or carry != 0:
        if l1 is not None:
            carry += l1.val
            l1 = l1.next
        if l2 is not None:
            carry += l2.val
            l2 = l2.next
        dummy.next = Node(carry%10)
        carry = carry/10

        dummy = dummy.next

    return result.next

def word_break(wordDict:List[str],s:str):
    def dfs(index):
        if index == len(s):
            return True
        for word in wordDict:
            if s[index:].startswith(word):
                if dfs(index+len(word)):
                    return True

        return False

    return dfs(0)

def generate_parenthesis(n:int):
    res = []
    def dfs(path,open,close):
        if open == close == n:
            res.append("".join(path))
            return
        if open < n:
            path.append("(")
            dfs(path,open+1,close)
            path.pop()
        if close < open:
            path.append(")")
            dfs(path, open, close+1)
            path.pop()

    dfs([],0,0)
    return res

def number_of_islands(grid):
       
    def dfs(grid,row,col):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == "0":
            return 
        
        grid[row][col] = "0"
        dfs(grid,row+1,col)
        dfs(grid,row-1,col)
        dfs(grid,row,col+1)
        dfs(grid,row,col-1)
        
    count = 0
    
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1":
                count +=1
                dfs(grid,row,col)

    return count 

def max_product_subarray(nums):
    res = max(nums)
    current_min, current_max = 1,1
    for num in nums:
        if num == 0:
            current_max,current_min = 1,1 
        
        temp = num*current_max
        
        current_max = (num*current_max,num*current_min,num)
        current_min = (temp,num*current_min,num)
        
        res = max(current_max,res)
    return res 

def find_min_in_rotated_sorted_array(nums:List[int]):
    boundary_index = -1 
    left =0 
    right = len(nums)-1
    
    while left <= right:
        mid = (left+right)//2
        
        if nums[mid]<= nums[-1]:  # confirm if array is still sorted 
            boundary_index = mid 
            right = mid-1 
        else:
            left = mid+1
    return boundary_index

def missing_numbers(nums:List[int]):
    myset = set(nums)
    
    for i in range(len(nums)+1):
        if i not in myset:
            return i
        

def anagaram(s1:str,s2:str):
    priority_queue = []
    
    for char in s1:
        heappush(priority_queue,char)
    
    for i in range(len(s2)):
        char = s2[i]
        while char in priority_queue:
            heappop(priority_queue)
    
    #print(priority_queue)
    if len(priority_queue) == 0:
        return True 
    else:
        return False 
def climbing_stairs(n):
     if n <=3:
         return n 
     dp =[0,1,2]
     
     for i in range(3,n+1):
        dp.append(dp[i-1]+dp[i-2])
         
     return dp[-1]  

def coin_change(coins,amounts):
    dp = [inf]*(amounts+1)
    dp[0] = 0 
    
    for amount in range(1 ,amounts+1):
        for coin in coins: 
            if amount-coin >=0:
                dp[amount] = min(dp[amount-coin]+1,dp[amount])
    
    return dp[-1] if dp[-1] < inf else -1 

    
def longest_increasing_subsequence(nums):
     dp = [1]*len(nums)
     best = 0 
     for i in range(len(nums)):
         for j in range(i):
            if nums[j] >= nums[i]: 
                 continue
            dp[i] = max(dp[i],1+dp[j])
         best = max(best,dp[i])
     return best 
    
def search_in_rotated_array(nums):
    pass  

def three_sum(nums,target):
    pass  

def number_one_bits():
    pass 

def counting_bits():  
    pass    

def search_in_rotated_sorted_array(nums:List[int]):
    pass 
    
def reverse_bits():
    pass    

class MedianOfStream:
    
    def __init__(self) -> None:
        pass
    def add_number(self,num:float):
        pass 
    def get_median():
        pass 




