import heapq


def a_star_search(graph, heuristics, start, goal):
    
    open_set = [(heuristics[start], start, [start], 0)]
    visited = {}

    while open_set:
        f_score, current, path, g_score = heapq.heappop(open_set)

       
        if current == goal:
            return path, g_score

        
        if current in visited and visited[current] <= g_score:
            continue
        visited[current] = g_score

        
        for neighbor, weight in graph.get(current, []):
            tentative_g = g_score + weight
            h_score = heuristics.get(neighbor, 0)
            total_f = tentative_g + h_score

           
            heapq.heappush(
                open_set, (total_f, neighbor, path + [neighbor], tentative_g)
            )

    return None, float("inf")



    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2), ("D", 5)],
    "C": [("D", 1)],
    "D": [],
}


heuristics = {"A": 5, "B": 3, "C": 1, "D": 0}


path, cost = a_star_search(graph, heuristics, "A", "D")
print(f"Path: {path}, Cost: {cost}")