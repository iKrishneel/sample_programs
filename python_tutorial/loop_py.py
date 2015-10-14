#!/usr/bin/python

Money = 2000
def addMoney():
    global Money
    Money += 2

def main():
    print Money
    addMoney()
    print Money

if __name__ == '__main__':
    main()

