import Queue

from item import Item
from point import Point


class TreeNode():

    def __init__(self, point, matrix, cost):
        self.point = point
        self.matrix = matrix
        self.cost = cost
        self.child_list = []
        self.level = 0
        self.id = 0

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_level(self, level):
        self.level = level

    def get_level(self):
        return self.level

    def add_child(self, TreeNode):
        self.child_list.append(TreeNode)

    def delete_child(self, TreeNode):
        self.child_list.remove(TreeNode)

    def get_child(self):
        return self.child_list

    def get_point(self):
        return self.point

    def get_matrix(self):
        return self.matrix

    def get_cost(self):
        return self.cost

    def __attrs(self):
        return (self.point, self.cost, self.level, self.id)

    def __cmp__(self, other):
        if(self.cost == other.cost):
            return -cmp(self.level, other.level)
        return cmp(self.cost, other.cost)

    def __eq__(self, other):
        return self.id == other.id and self.matrix == other.matrix

    def __hash__(self):
        return hash(self.__attrs())

    def __str__(self):
        return 'Node(%s, %s) @ level %s' % (self.point.get_x(), self.point.get_y(), self.level)


if __name__ == "__main__":
    mat = [[1,2,3], [2,3,4], [3,4,5]]
    point1 = Point(2,3)
    point2 = Point(4, 5)
    a = TreeNode(point1, mat, 1)
    b = TreeNode(point2, mat, 0)

    queue = Queue.PriorityQueue()
    queue.put(a)
    queue.put(b)
    print (queue.qsize())
    print (queue.get())
    print (queue.qsize())

    paths = {}
    paths[a] = 1
    paths[c] = 2

    print paths[b]

    # myTree = ['a', ['b', ['d', [], []], ['e', [], []]], ['c', ['f', [], []], []]]
    # print(myTree)
    # print('left subtree = ', myTree[1])
    # print('root = ', myTree[0])
    # print('right subtree = ', myTree[2])

    # print a.get_matrix()
