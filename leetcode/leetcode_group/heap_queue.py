from typing import Counter, List 
from heapq import heappop,heappush
def K_closest_points(points,k:int):
    priority_queue = []
    results = []
    
    for sub_array in points:
        x = sub_array[0]
        y = sub_array[1]
        
        distance = -(x*x + y*y)
        heappush(priority_queue,[distance,sub_array])
        
        if len(priority_queue) > k: 
            heappop(priority_queue)
        

    for item in priority_queue:
        results.append(item[1])

    return results

def kth_largest_element_in_array(keys:List[int],k:int)->int: 
    priority_queue = []
    for key in keys:
        heappush(priority_queue,key)
        if len(priority_queue) > k:
            heappop(priority_queue)

    print(priority_queue)
    return priority_queue[0]

def top_k_frequent_elements(keys:List[int],k:int):
    priority_queue = []
    dic = Counter(keys)
    results = []
    
    for key in dic.keys():
        heappush(priority_queue,[dic[key],key])
        if len(priority_queue) >k:
            heappop(priority_queue)
    
    print(priority_queue)        
    for item in priority_queue:
        results.append(item[1])
    
    return results 

def top_most_frequent_words(words:List[str],k:int):
    dic = Counter(words)
    results = []
    priority_queue = []
    
    for key in dic.keys():
        heappush(priority_queue,[dic[key],key])
        if len(priority_queue) >k:
            heappop(priority_queue)
    
    print(priority_queue)        
    for item in priority_queue:
        results.append(item[1])
    
    return results 
    

if __name__ == '__main__':
   keys = [1,1,1,2,2,3]
   print(top_k_frequent_elements(keys,2))