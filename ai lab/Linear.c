#include <stdio.h>

int linear_search_iterative(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int linear_search_recursive(int arr[], int size, int index, int target) {
    if (index >= size) {
        return -1;
    }
    if (arr[index] == target) {
        return index;
    }
    return linear_search_recursive(arr, size, index + 1, target);
}

int main() {
    int arr[] = {10, 25, 30, 45, 50, 65, 80};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target;

    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Enter number to search: ");
    scanf("%d", &target);

    int result_iterative = linear_search_iterative(arr, size, target);
    int result_recursive = linear_search_recursive(arr, size, 0, target);

    printf("\nSearch results:\n");
    printf("Using iteration: ");
    if (result_iterative != -1) {
        printf("Found at index %d\n", result_iterative);
    } else {
        printf("Not found\n");
    }

    printf("Using recursion: ");
    if (result_recursive != -1) {
        printf("Found at index %d\n", result_recursive);
    } else {
        printf("Not found\n");
    }

    return 0;
}