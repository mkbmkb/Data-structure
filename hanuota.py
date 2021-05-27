# -*- coding: utf-8 -*-
#!/bin/env python
'''
count = 0
def hanoi(n,x,y,z):
    if n == 1:
        global count
        count += 1
        print(count,':',x,'-->',z)
    else:
        hanoi(n-1,x,z,y)
        count += 1
        print(count, ':', x, '-->', z)
        hanoi((n-1),y,x,z)

hanoi(4,'1','2','3')
'''
#栈解法

class Stack():
    def __init__(self):
        self.list = list()

    def push(self,item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def __str__(self):
        return  str(self.list)


time = 0

def HanoiTower(n,a,b,c):
    global time
    if n == 0:
        return

    HanoiTower(n - 1,a,c,b)
    c.push(a.pop())
    time += 1
    HanoiTower(n - 1,b,a,c)

def main():
    a = Stack()
    b = Stack()
    c = Stack()
    count = 2
    for i in range(count):
        a.push(i+1)
    print(a)
    print(b)
    print(c)
    print('====')
    HanoiTower(count,a,b,c)
    print(a)
    print(b)
    print(c)
    print('times',time)


if __name__ == '__main__':
    main()



