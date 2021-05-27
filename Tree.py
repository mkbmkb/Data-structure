class BinaryTree:
    def __init__(
        rootObj):
    self.key = rootObj
    self.leftChild = None
    self.rightChild = None


def insertLeft(self, newNode):
    if  self.rightChild == None:
        self.leftChild == BinaryTree(newNode)

    else:
        t = BinaryTree(newNode)
        t.rightChild = self.rightChild
        self.rightChild = t


