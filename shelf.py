from item import Item
from point import Point


class Shelf():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.items = []
        self.size = 0
        self.left = Point(x - 1, y)
        self.right = Point(x + 1, y)

    def put(self, item):
        self.items.append(item)
        self.size += 1

    def has_item(self, item):
        return item in self.items

    def get_items(self):
        return self.items

    def remove(self, item):
        self.items.remove(item)
        self.size -= 1

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_point(self):
        return Point(self.x, self.y)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def size(self):
        return self.size

    def __attrs(self):
        return (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.__attrs())

    def __str__(self):
        return 'Shelf @(%s, %s) contains %s items' % (self.x, self.y, self.size)


if __name__ == "__main__":
    s1 = Shelf(1, 1)
    s2 = Shelf(1, 1)
    a = Item('a', 1, 1)
    b = Item('b', 1, 1)

    s1.put(a)
    s1.put(b)

    print s1.get_right()
