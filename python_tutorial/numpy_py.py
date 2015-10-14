from numpy import *

def main():
    print 'Executing main program....\n'
    a = arange(15).reshape(3,5)
    print a

    b = array([20, 30, 40, 50])
    
    print a.shape
    print a.dtype.name
    print a.itemsize
    print a.size
    print b

    
if __name__ == '__main__':
    main()
