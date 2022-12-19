def parse_num(num):
    # parse string represenation of float, such that
    # it is rounded at most two digits
    # but only to non-zero decimal places
    # parse float
    if float(num).is_integer():
        return str(int(num))
    else:
        ret = str(round(float(num), 2))
        decimals = str(round(float(num), 2)).split(".")[1]
        for i in range(2 - len(decimals)):
            ret += "0"
        return ret