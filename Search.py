#冒泡排序
alist = [54,26,12,93,1123,17,3,77,31,44,55,20,111]

def bubbleSort(alist):
    exchanges = True
    passnum = len(alist) - 1
    while passnum > 0:
        exchanges = False
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                exchanges = True
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
        passnum = passnum -1

def insertionSort(alist):
    for index in range(1,len(alist)):
        currentvalue = alist[index]
        position = index

        while position > 0 and  alist[position-1] > currentvalue:
            alist[position] = alist[position-1]
            position = position - 1
            alist[position] = currentvalue

k = 0
#谢尔排序
def gapinsertionSort(alist,start,gap):
    for index in range(start+gap,len(alist)):
        currentvalue = alist[index]
        position = index

        while position >= gap  and  alist[position-gap] > currentvalue:
            alist[position] = alist[position-gap]
            position = position - gap
            alist[position] = currentvalue

def shellSort(alist):
    sublistcount = len(alist)//2

    while sublistcount > 0:
        for staticmethod in range(sublistcount):
            gapinsertionSort(alist,staticmethod,sublistcount)
        print('after increments of size',sublistcount,'the list is',alist)
        sublistcount = sublistcount//2

# shellSort(alist)

#归并
def merge_sort(alist):
    n = len(alist)
    if n <= 1: #嵌套底部
        return alist
    mid = n//2

    left = alist[:mid]
    right = alist[mid:]

    left_li = merge_sort(left)
    right_li = merge_sort(right)

    # merge(left,right)
    left_pointer,right_pointer = 0,0
    result = []
    while left_pointer < len(left_li) and right_pointer < len(right_li):
        if left_li[left_pointer]  < right_li[right_pointer]:
            result.append(left_li[left_pointer])
            left_pointer += 1

        else:
            result.append(right_li[right_pointer])
            right_pointer += 1

    result += left_li[left_pointer:]
    result += right_li[right_pointer:]

    return result
