# Bubble Sort - AI Generated Style
# This algorithm sorts a list by repeatedly swapping adjacent elements if they are in the wrong order.

def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):
        swapped = False

        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Swap if element is greater than next
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swap occurred, array is sorted
        if not swapped:
            break

    return arr

# Test the implementation
if __name__ == "__main__":
    numbers = [64, 34, 25, 12, 22, 11, 90]
    print("Original:", numbers)
    sorted_numbers = bubbleSort(numbers.copy())
    print("Sorted:  ", sorted_numbers)