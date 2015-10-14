#include <iostream>
using namespace std;

class Square;

class Rectangle {
    int width, height;
  public:
    int area ()
      {return (width * height);}
    void convert(Square a);

 private:
    int x;
};

class Square:public Rectangle {
   friend class Rectangle;
   private:
     int side;
   public:
   explicit Square(int a) : side(a) {x = 19;}
};

void Rectangle::convert(Square a) {
  width = a.side;
  height = a.side;
}
  
int main() {
  Rectangle rect;
  Square sqr(4);
  sqr.convert(sqr);
  std::cout << sqr.area() << std::endl;
  
  rect.convert(sqr);
  cout << rect.area() << "\n";
  return 0;
}
