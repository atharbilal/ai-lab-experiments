# Simple Hill Climbing Algorithm

def f(x):
    return -(x - 5) ** 2 + 25


def hill_climbing(start):
    current = start

    while True:

        # Generate neighbors
        left = current - 1
        right = current + 1

        # Move to a better neighbor
        if f(left) > f(current):
            current = left

        elif f(right) > f(current):
            current = right

        else:
            # No better neighbor
            break

    return current, f(current)


# Starting point
start = 0

solution, value = hill_climbing(start)

print("Starting Point:", start)
print("Best Solution:", solution)
print("Maximum Value:", value)