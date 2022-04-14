from typing import List

def remove_duplicates(arr: List[int]) -> int:
    
    count = 0
    window_start =0 
    
    for i in range(len(arr)):
        if arr[window_start] != arr[i]:
            count +=1
            window_start = i 
        continue
    return count+1


if __name__ == '__main__':
    nums = [0,0,1,1,1,2,2]
    print(remove_duplicates(nums))
   
