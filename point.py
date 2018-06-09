class Point():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_x(self, value):
        self.x = value

    def set_y(self, value):
        self.y = value

    def __str__(self):
        return '(%s, %s)' % (self.x, self.y)

    def __attrs(self):
        return (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # def __ne__(self, other):
    #     return self.x != other.x or self.y == other.y

    def __hash__(self):
        return hash(self.__attrs())


if __name__ == "__main__":
    a = Point(12, 22)
    b = Point(12, 23)
    print (a == b)