class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

data1 = {'cost': 1, 'solution': "East1", "constraints": []}
data2 = {'cost': 2, 'solution': "East2", "constraints": []}
data3 = {'cost': 3, 'solution': "East3", "constraints": []}
r = BinaryTree(data1)
print"root of tree: ", (r.getRootVal())
#print(r.getLeftChild())
r.insertLeft(data2)
#print (r.getLeftChild())
print "left child: ", (r.getLeftChild().getRootVal())
r.insertRight(data3)
#print(r.getRightChild())
print "right child", (r.getRightChild().getRootVal())
#r.getRightChild().setRootVal('hello')
#print(r.getRightChild().getRootVal())
