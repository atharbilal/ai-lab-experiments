#include <stdio.h>
#define MAX 100
int visited[MAX];
void dfs(int graph[MAX][MAX], int n, int start)
{
    visited[start] = 1;
    printf("%d ", start);
    for (int i = 0; i < n; i++)
    {
        if (graph[start][i] && !visited[i])
        {
            dfs(graph, n, i);
        }
    }
}
int main()
{
    int n = 4;
    int graph[MAX][MAX] = {
        {0, 1, 1, 0},
        {1, 0, 0, 1},
        {1, 0, 0, 1},
        {0, 1, 1, 0}};
    for (int i = 0; i < MAX; ++i)
    {
        visited[i] = 0;
    }
    printf("\nDFS Traversal: ");
    dfs(graph, n, 0);
    return 0;
}