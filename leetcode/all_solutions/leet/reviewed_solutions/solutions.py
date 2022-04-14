def exist(board,word):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == word[0] and dfs(board,row,col,word): 
                return True 
            
    return False 

def dfs(board,row,col,word):
        
        if len(word) ==0:
            return True 

        if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or board[row][col] != word[0]:
            return False 
        
        temp = board[row][col]
        board[row][col] = ' '
        
        if dfs(board,row+1,col,word[1:]) or dfs(board,row-1,col,word[1:]) or dfs(board,row,col+1,word[1:]) or dfs(board,row,col-1,word[1:]):
            return True 
        
        board[row][col] = temp 

        return False 

def productArrayExceptSelf(nums):
    results = [1]*len(nums)
    
    mul = 1 
    for i in range(len(nums)): 
        
        results[i] = results[i]*mul 
        mul = nums[i]*mul
        
        mul = 1 
    for i in range(len(nums)-1,-1,-1):  
           results[i] = results[i]*mul 
           mul = nums[i]*mul

    return results

def subArraySums(nums,target):
    dic = dict() 
    dic.setdefault(0,1)
    summ = 0  
    count = 0 
    
    for num in nums: 
        summ += num 
        
        temp = target -summ 
        if temp in dic:
            count += dic.get(temp)
        
        dic[summ] +=1 

    return count 

def rotateArray(nums,k): 
    
    def helper(arr,left,right): 
        
        while left < right : 
            
            arr[left], arr[right] = arr[right],arr[left]
            
            left +=1 
            right -=1 
    
    helper(nums,0,len(nums)-1)
    helper(nums,0,k-1)
    helper(nums,k,len(nums)-1)    

def maxProfit(prices): 
    cheapest_price = prices[0]
    profit = 0  
    
    for price in prices:  
        cheapest_price = min(cheapest_price,price)
        profit = max(profit,price-cheapest_price)
    
    return profit


class TreeNode: 
    
    def __init__(self,val,left=None,right=None): 
        
        self.val = val  
        self.left = left  
        self.right = right 
class Node: 
    
    def ___init__(self,val,next=None): 
        self.val = val 
        self.next = next

def searchBST(root:TreeNode,val):
    if root.val == val:  
        return root     
    
    elif root.val > val:  
        return searchBST(root.left,val)
    elif root.val < val :
        searchBST(root.right,val)

def addTwoNumbers(l1,l2): 
    
    dummy = Node(0)
    
    result = dummy 
    carry = 0 
    
    while l1 is not None or l2 is not None or carry !=0 :
        
        if l1 is not  None: 
            carry += l1.val
            
            l1 = l1.next
        
        if l2 is not None: 
            carry += l2.val
            l2 = l2.next

        dummy.next = Node(carry%10)
        carry = carry/10
        
        dummy = dummy.next 

    return result.next 

def climbingStairs(n): 

    if n <= 3: return n 
    
    dp = [0,1,2,3]
    for i in range(4,n+1):
        dp[i] = dp[i-1]+dp[i-2]

    return dp.pop()

def maxSubArraySum(nums):
    currentSum = nums[0]
    maxSum = nums[0]
    
    for i in range(1,len(nums)):
        currentSum = max(currentSum+nums[i],nums[i])
        maxSum = max(maxSum,currentSum)
    
    return maxSum 
    
    