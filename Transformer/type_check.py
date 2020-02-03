from argparse import ArgumentTypeError

def bool_type_check(string) :
    if string.lower() == 'true' :
        return True
    elif string.lower() == 'false' :
        return False
    else :
        raise ArgumentTypeError(
            "\n\tTrue 또는 False만 입력 가능 (입력된 값 : {})\n".format(string)
        )