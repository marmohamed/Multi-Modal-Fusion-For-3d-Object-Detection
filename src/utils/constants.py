def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

class Const(object):
    @constant
    def IMAGE_BRANCH():
        return 1
    @constant
    def BEV_BRANCH():
        return 2
    @constant
    def FV_BRANCH():
        return 3