

def ab(**kwargs):
    print(kwargs['a'])

def c(**kwargs):
    print(a)


di = dict(a=5, b=6)
di = {"a": 5, "b": 7}
ab(**di)
c(**di)
ab(a=5)