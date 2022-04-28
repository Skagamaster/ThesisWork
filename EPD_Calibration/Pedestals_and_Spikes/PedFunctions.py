

# This converts from indexed tile to [ew, pp, tt], as str.
def convert(pos=123):
    ew = int(pos/372)
    pp = int((pos-372*ew)/31)
    tt = (pos % 31) + 1
    pp = pp+1
    set = [ew, pp, tt]
    return set


def backConvert(set=[0, 3, 28]):
    pos = 372*set[0]+12*set[1]+set[2]
    return pos
