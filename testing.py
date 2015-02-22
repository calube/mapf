class CTNode:
    left , right, data = None, None, 0
    # data = {'cost':0, 'constraints':0, 'solution':[]}
    
    def __init__(self, data):
        # initializes the data members
        self.left = None
        self.right = None
        self.data = data

class ConstraintTree:
    def __init__(self):
        # initializes the root member
        self.root = None
    
    def addNode(self, data):
        # creates a new node and returns it
        return CTNode(data)

    def insert(self, root, data):
        # inserts a new data
        if root == None:
            # it there isn't any data
            # adds it and returns
            return self.addNode(data)
        else:
            # enters into the tree
            if data['cost'] <= root.data['cost']:
                # if the data is less than the stored one
                # goes into the left-sub-tree
                root.left = self.insert(root.left, data)
            else:
                # processes the right-sub-tree
                root.right = self.insert(root.right, data)
            return root
        
    def lookup(self, root, target):
        # looks for a value into the tree
        if root == None:
            return 0
        else:
            # if it has found it...
            if target == root.data:
                return 1
            else:
                if target < root.data:
                    # left side
                    return self.lookup(root.left, target)
                else:
                    # right side
                    return self.lookup(root.right, target)
        
    def minValue(self, root):
        # goes down into the left
        # arm and returns the last value
        while(root.left != None):
            root = root.left
        return root.data

    def maxDepth(self, root):
        if root == None:
            return 0
        else:
            # computes the two depths
            ldepth = self.maxDepth(root.left)
            rdepth = self.maxDepth(root.right)
            # returns the appropriate depth
            return max(ldepth, rdepth) + 1
            
    def size(self, root):
        if root == None:
            return 0
        else:
            return self.size(root.left) + 1 + self.size(root.right)

    def printTree(self, root):
        # prints the tree path
        if root == None:
            pass
        else:
            self.printTree(root.left)
            print root.data,
            self.printTree(root.right)

    def printRevTree(self, root):
        # prints the tree path in reverse order
        if root == None:
            return
        else:
            self.printRevTree(root.right)
            print root.data,
            self.printRevTree(root.left)

if __name__ == "__main__":
    # create the binary tree
    CT = ConstraintTree()

    # add the root node
    data = {'cost':[1,2,3]}
    root = CT.addNode(data)
    depth = CT.maxDepth(root)
    print "depth after adding 1 node: ", depth
    print "printing tree: ", CT.printTree(root)
    print "\n \n"
    # insert values
    data1 = data['cost']
    data1.append(4)
    test = data1
    data1 = {'cost': test}
    print "data1: ", data1
    CT.insert(root, data1)
    #CT.addNode(data1)
    depth = CT.maxDepth(root)
    print "depth after adding 2 nodes: ", depth
    print "printing tree: ", CT.printTree(root)
    print "\n \n"
    data2 = data['cost']
    data2.append(5)
    print "data2: ", data2
    test1 = data2
    data2 = {'cost': test1}
    CT.insert(root, data2)
    #CT.addNode(data2)
    depth = CT.maxDepth(root)
    print "depth after adding 3 nodes: ", depth
    print "printing tree: ", CT.printTree(root)
    print "\n \n"
    data3 = {'cost':4}
    CT.insert(root, data3)
    #CT.addNode(data3)
    depth = CT.maxDepth(root)
    print "depth after adding 4 nodes: ", depth
    print "printing tree: ", CT.printTree(root)
    print "\n \n"
    data4 = {'cost':5}
    CT.insert(root, data4)
    #CT.addNode(data4)
    depth = CT.maxDepth(root)
    print "depth after adding 5 nodes: ", depth
    print "printing tree: ", CT.printTree(root)
    print "\n \n"



    
    
    print "printing tree: ", CT.printTree(root)
    print "printing tree in reverse: ", CT.printRevTree(root)      
    print "min value of tree: ", CT.minValue(root)
    print "max value of tree: ", CT.maxDepth(root)
    print "size of tree: ", CT.size(root)