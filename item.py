from point import Point


class Item():

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __str__(self):
        return 'Item %s @(%s, %s)' % (self.id, self.x, self.y)



if __name__ == "__main__":
    a = Item(123, 12, 22)
    print a
