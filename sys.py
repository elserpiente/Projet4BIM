import sys
if sys.maxsize > 2**32:
    print("%x" % sys.maxsize,"64 bit")
else:
    print("%x" % sys.maxsize,"32 bit")
