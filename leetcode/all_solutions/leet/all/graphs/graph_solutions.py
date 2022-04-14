"""
How to approach Graph questions:

Undirected Graph :


"""


def countConnectedComponents(N, edges):
    adjacency_list = {}
    visited = {}
    count = 0

    for vertex in range(N):
        adjacency_list[vertex] = []
        visited[vertex] = False

    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]
        adjacency_list[v1].append(v2)
        """  We append v1 and v2 since its undirected graph """
        adjacency_list[v2].append(v1)

    def dfs(vertex):
        visited[vertex] = True

        for neighbor in adjacency_list[vertex]:
            if not visited[vertex]:
                dfs(neighbor)

    for vertex in range(N):
        if not visited[vertex]:
            dfs(vertex)
            count += 1

    return count


def hasCycle(N, edges):
    """
    Detect a cycle in a directed graph :

    White   --- Not Visited
    Gray    --- Visiting
    Black   ---- Visited
    Cycle is detected if two grays are connected

    initialize the visited nodes to white- since its not visited
    loop through the vertex
    set the vertex to gray while visiting
    check if the neigbors are gray - return true - it has a cycle

    check for the neighbors that are white, dfs them and return True if gray connected

    then set the rest of the nodes to black since we already visited all gray nodes


    check for all nodes and dfs around them



    """

    adjacent_list = {}
    visited = {}

    for vertex in range(N):
        adjacent_list[vertex] = []
        visited[vertex] = "white"  # not visited yet

    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]

        adjacent_list[v1].append(v2)  # directed graph - only ones appended

    def dfs(vertex):

        visited[vertex] = "gray"  # visiting

        for neighbor in adjacent_list[vertex]:
            if visited[neighbor] == "gray":
                return True

            if visited[neighbor] == "white" and dfs(neighbor):
                return True

        visited[vertex] = "black"  # visited
        return False

    for vertex in range(N):
        if dfs(vertex):
            return True

    return False


def canFinish(self, numCourses, prerequisites):
    adjacent_list = {}
    visited = {}

    for vertex in range(numCourses):
        adjacent_list[vertex] = []
        visited[vertex] = "white"  # not visited yet

    for edge in prerequisites:
        v1 = edge[0]
        v2 = edge[1]
        adjacent_list[v1].append(v2)

    def dfs(vertex):

        visited[vertex] = "gray"  # visiting

        for neighbor in adjacent_list[vertex]:
            if visited[neighbor] == "gray":
                return True
            if visited[neighbor] == "white" and dfs(neighbor):
                return True

        visited[vertex] = "black"
        return False

    for vertex in range(numCourses):
        if visited[vertex] == "white":
            if dfs(vertex):
                return False

    return True
