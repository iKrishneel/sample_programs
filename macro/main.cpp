
#include <iostream>

#define PRINT_RED(X) \
    ((std::cout<< "\033[1m\033[35m" << X << " \033[0m" << std::endl), \
    (void) 0)\

int main(int argc, char *argv[]) {
   
    PRINT_RED("Hello World");
    PRINT_RED(2);
    
    return 0;
}
