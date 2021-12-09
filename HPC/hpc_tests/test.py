import sys
print ("the script has the name %s" % (sys.argv[0]))
if len(sys.argv) > 1:
    print ("the script has arguments %s" % (sys.argv[1:]))