#!/usr/bin/python

import sys


# An example of a class
class Shape:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        description = "This shape has not been declared"
        author = "Nobody is author"

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return 2*self.x + 2*self.y

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x *= scale
        self.y *= scale

# class inheritance
class Square(Shape):
    def __init__(self, x):
        self.x = x
        self.y = x

class Triangle(Shape):
    def area(self):
        return 0.5 * self.x * self.y
    
def main(option):
    print 'Executing main...'
    if option:
        rectangle = Shape(100, 45)
        print rectangle.area()
    else:
        triangle = Triangle(40, 20)
        print triangle.area()
        #square = Square(20)
        #print square.perimeter()

        
if __name__ == '__main__':
    main(False)
else:
    print 'Running Square....'
    square = Shape(20, 20)
    print square.perimeter()
