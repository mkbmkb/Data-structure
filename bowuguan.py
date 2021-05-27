tr = [{'v':0,'w':0},{'v':3,'w':2},{'v':4,'w':3},{'v':5,'w':4},{'v':6,'w':5}]
max_w = 8
m  = [[0]*(max_w +1) for i in range(len(tr)+1)]
# m = {(i,w):0 for i in range(len(tr)) for w in range(max_w + 1)}

for i in range(1,len(tr)): #行
    for w in range(1,max_w+1): #列

       if tr[i]['w'] > w:
           m[i][w] = m[i-1][w]
       else:
           m[i][w] = max(m[i-1][w],m[i-1][w - tr[i]['w']] + tr[i]['v'])








