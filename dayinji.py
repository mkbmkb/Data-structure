# -*- coding: utf-8 -*-
#!/bin/env python
'''
问题：1小时内，10个学生。每人打印2次，打印1次1-20页不等。打印机打印10页/分钟或者5页、分钟。

思路：
打印任务的属性：提交时间、打印页数
打印队列的属性：FIFO队列性质
打印机的属性：打印速度、是否忙

生成作业概率：20/1 hours = 1/180s

'''

from queue import Queue
import random

class Printer:
    def __init__(self,ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = 0

    def trick(self):
        if self.currentTask != None:
            self.timeRemaining = self.timeRemaining - 1 #打印一秒
            if self.timeRemaining <= 0:
                self.currentTask = None   #进入空闲

    def busy(self):
        if self.currentTask != None:
            return True
        else:
            return False

    def startNext(self,newtask):
        self.currentTask = newtask
        self.timeRemaining  = newtask.getPages()*60/self.pagerate  #计算需要打印多久 #引入新参数

class Task:
    def __init__(self,time):
        self.timestamp = time  #生成时间
        self.pages = random.randrange(1,21)

    def getStamp(self):
        return self.timestamp

    def getPages(self):
        return self.pages

    def waitTime(self,currenttime):
        return currenttime - self.timestamp   #引入新参数

def newPrintTask():
    num = random.randrange(1,181)
    if num == 180:
        return True
    else:
        return False

def simulation(numSeconds,pagesPerMinute): #多长时间以及打印机速度
    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []

    for currentSecond in range(numSeconds):#消耗时间
        if newPrintTask():
            task = Task(currentSecond) #生成时间作为参数
            printQueue.enqueue(task) #入队

        if (not labprinter.busy()) and (not printQueue.isEmpty()): #打印机空闲而且打印队列中有作业
            nexttask = printQueue.dequeue()
            waitingtimes.append(nexttask.waitTime(currentSecond)) #把等待时间放在等待时间列表里
            labprinter.startNext(nexttask)

        labprinter.trick() #打印一秒

    averageWait = sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining." %(averageWait,printQueue.size()))
for i in range(10):
    simulation(3600,5)





