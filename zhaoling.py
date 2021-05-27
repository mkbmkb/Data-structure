
#递归法
# def recMC(coinValueList,change,knowResults):
#     if change in coinValueList:
#         knowResults[change] = 1
#         return 1
#     elif knowResults[change] > 0:
#         return knowResults[change]
#     else:
#         for i in [c for c in coinValueList if c <= change]:
#             numCoins = 1 + recMC(coinValueList,change - i,knowResults)
#             knowResults[change] = numCoins
#
#     return numCoins
#
# print(recMC([1,5,10],12,[0]*13))

#动态规划
def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
    for cents in range(1,change+1):
        coinCount = cents
        newCoin = 1
        for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
                coinCount = minCoins[cents - j] + 1
                newCoin = j

            minCoins[cents] = coinCount
            coinsUsed[cents] = newCoin

    return minCoins[change]

def printCoins(coinsUsed,change):
    coin = change
    while coin > 0:
        thisCoin = coinsUsed[coin]
        print(thisCoin)
        coin = coin - thisCoin

amnt = 11
clist = [1,5,10]
coinsUsed = [0] * (amnt + 1)
coinCount = [0] * (amnt + 1)

print(dpMakeChange(clist,amnt,coinsUsed,coinCount))
printCoins(coinsUsed,amnt)
print(coinsUsed)

