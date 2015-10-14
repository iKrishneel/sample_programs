#!/usr/bin/env python

import sys
import getopt

def main(argv):
    # if len(argv) == 1:
    #     print "too few arguments"
    #     print 'test.py -i <inputfile> -o <outputfile>'
    #     return
    
    input_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    print 'INPUT ', input_file
    print 'OUT ', output_file
        

if __name__ == "__main__":
    main(sys.argv[1:])
