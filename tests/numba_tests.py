import numpy as np
from numba import njit, int64, jitclass
import time
import numpy as np
from numba import njit


representation = np.zeros((10, 10), dtype=np.bool_)
changed_lines = np.array([1, 2], dtype=np.int64)
lowest_free_rows = np.zeros(10, dtype=np.int64)
num_columns = np.int64(10)


"""
Advanced list indexing
"""

@njit
def with_integers():
    l = [1, 2, 3]

    return l[np.array([True, False, True])]

with_integers()


"""
update bool with 1s

WEIRD: 1s seem to be slightly faster than Trues...

"""

@njit
def update_with_bool():
    arr = np.zeros(100000000, dtype=np.bool_)
    for ix in range(len(arr)):
        arr[ix] = True
    return arr

@njit
def update_with_one():
    arr = np.zeros(100000000, dtype=np.bool_)
    for ix in range(len(arr)):
        arr[ix] = 1
    return arr

s = time.time()
update_with_bool()
e = time.time()
print(e-s)

s = time.time()
update_with_one()
e = time.time()
print(e-s)


"""
enumerate vs range


result: sorta equal...
"""

arr = np.random.random(100000000)
@njit
def f_enumerate(arr):
    a = 0
    for ix, con in enumerate(arr):
        a += con
    return a

@njit
def f_range(arr):
    a = 0
    for ix in range(len(arr)):
        con = arr[ix]
        a += con
    return a

print("out", f_enumerate(arr))
print("out", f_range(arr))

s = time.time()
f_enumerate(arr)
e = time.time()
print(e-s)

s = time.time()
f_range(arr)
e = time.time()
print(e-s)



"""
No argument names.... need all optional arguments!
"""
@njit
def np_arange():
    a = np.arange(1, 5, 1, np.int64)
    return a

np_arange()


"""
BAD MEMORY ACCESS
"""
import numpy as np
from numba import njit

# Expected behavior in numpy
def access_out_of_bounds_index():
    arr = np.zeros(5)
    arr[0] = 1
    arr[4] = 1
    arr[5] = 1  # Error index out ouf bounds
    return arr

access_out_of_bounds_index()


@njit
def JITTED_access_out_of_bounds_index():
    arr = np.zeros(5)
    arr[0] = 1
    arr[4] = 1
    arr[5] = 1  # Error index out ouf bounds
    return arr

JITTED_access_out_of_bounds_index()



num = 10000000

np.random.seed(1)
b_ = np.random.randint(0, 2, num, dtype=np.bool_)
np.random.seed(1)
b = np.random.randint(0, 2, num, dtype=np.bool)
np.random.seed(1)
i_ = np.random.randint(0, 2, num, dtype=np.int_)
np.random.seed(1)
i64 = np.random.randint(0, 2, num, dtype=np.int64)
np.random.seed(1)
i32 = np.random.randint(0, 2, num, dtype=np.int32)
np.random.seed(1)
i8 = np.random.randint(0, 2, num, dtype=np.int64)


"""
Check for a lot of True/Falses...

Result:
bools are twice as fast as integers,

"""

@njit
def check_a_lot(input):
    a = 0
    for i in input:
        if i:
            a += 1
    return a



print("out", check_a_lot(b_))
print("out", check_a_lot(b))
print("out", check_a_lot(i_))
print("out", check_a_lot(i64))
print("out", check_a_lot(i32))
print("out", check_a_lot(i8))

s = time.time()
check_a_lot(b_)
e = time.time()
print(e-s)

s = time.time()
check_a_lot(b)
e = time.time()
print(e-s)

s = time.time()
check_a_lot(i_)
e = time.time()
print(e-s)

s = time.time()
check_a_lot(i64)
e = time.time()
print(e-s)

s = time.time()
check_a_lot(i32)
e = time.time()
print(e-s)

s = time.time()
check_a_lot(i8)
e = time.time()
print(e-s)


"""
Instead of checking 
with 
if element:

...check with
if element == 1

Result:
Only np.int64 is reasonably fast

"""

# @njit
# def check_a_lot_2(input):
#     a = 0
#     for i in input:
#         if i == 1:
#             a += 1
#     return a
#
#
# print("out", check_a_lot_2(b_))
# print("out", check_a_lot_2(b))
# print("out", check_a_lot_2(i_))
# print("out", check_a_lot_2(i64))
# print("out", check_a_lot_2(i32))
# print("out", check_a_lot_2(i8))
#
# s = time.time()
# check_a_lot_2(b_)
# e = time.time()
# print(e-s)
#
# s = time.time()
# check_a_lot_2(b)
# e = time.time()
# print(e-s)
#
# s = time.time()
# check_a_lot_2(i_)
# e = time.time()
# print(e-s)
#
# s = time.time()
# check_a_lot_2(i64)
# e = time.time()
# print(e-s)
#
# s = time.time()
# check_a_lot_2(i32)
# e = time.time()
# print(e-s)
#
# s = time.time()
# check_a_lot_2(i8)
# e = time.time()
# print(e-s)



"""
Summing bools / vs ints

"""


num = 100000000

np.random.seed(1)
b_ = np.random.randint(0, 2, num, dtype=np.bool_)
np.random.seed(1)
b = np.random.randint(0, 2, num, dtype=np.bool)
np.random.seed(1)
i_ = np.random.randint(0, 2, num, dtype=np.int_)
np.random.seed(1)
i64 = np.random.randint(0, 2, num, dtype=np.int64)
np.random.seed(1)
i32 = np.random.randint(0, 2, num, dtype=np.int32)
np.random.seed(1)
i8 = np.random.randint(0, 2, num, dtype=np.int64)

@njit
def sum_a_lot(input):
    acc = np.bool_(0)
    for i in input:
        acc += i
    return acc

# print("out", sum_a_lot(b_))
# print("out", sum_a_lot(b))
# print("out", sum_a_lot(i_))
# print("out", sum_a_lot(i64))
# print("out", sum_a_lot(i32))
# print("out", sum_a_lot(i8))

s = time.time()
sum_a_lot(b_)
e = time.time()
print(e-s)

s = time.time()
sum_a_lot(b)
e = time.time()
print(e-s)

s = time.time()
sum_a_lot(i_)
e = time.time()
print(e-s)

s = time.time()
sum_a_lot(i64)
e = time.time()
print(e-s)

s = time.time()
sum_a_lot(i32)
e = time.time()
print(e-s)

s = time.time()
sum_a_lot(i8)
e = time.time()
print(e-s)





'''
Experiment below:

Better NOT to update attributes of jitclass-method.
Rather define variable in method. Outsourcing completely is not needed.

'''

spec = [
    ('acc', int64),
    ('bla', int64)
]


@jitclass(spec)
class AllInClass:
    def __init__(self):
        self.acc = 0
        self.bla = 0

    def count_a_lot(self):
        self.bla = 2
        for i in range(50000000):
            self.acc += 1
        return self.acc


@jitclass(spec)
class NoAttributeClass:
    def __init__(self):
        self.bla = 0

    def count_a_lot(self):
        acc = 0
        self.bla = 2
        for i in range(50000000):
            acc += 1
        return acc


class OutSourceClass:
    def __init__(self):
        self.bla = 0

    def count_a_lot(self):
        count_a_lot()



@njit
def count_a_lot():
    acc = 0
    for i in range(50000000):
        acc += 1
    return acc


all_in_class = AllInClass()
no_attribute_class = NoAttributeClass()
out_source_class = OutSourceClass()

s = time.time()
all_in_class.count_a_lot()
e = time.time()
print(e-s)

s = time.time()
no_attribute_class.count_a_lot()
e = time.time()
print(e-s)

s = time.time()
count_a_lot()
e = time.time()
print(e-s)

s = time.time()
out_source_class.count_a_lot()
e = time.time()
print(e-s)




'''
Experiment same as above but with numpy arrays. Does one need to copy() ??

Better NOT to update attributes of jitclass-method.
Rather define variable in method. Outsourcing completely is not needed.

'''

spec = [
    ('acc', int64[:]),
    ('bla', int64)
]



@jitclass(spec)
class AllInClass:
    def __init__(self):
        self.acc = np.array([0])
        self.bla = 0

    def count_a_lot(self):
        self.bla = 2
        for i in range(1000000000):
            self.acc[0] += 1
        return self.acc


@jitclass(spec)
class ShallowCopyClass:
    def __init__(self):
        self.bla = 0
        self.acc = np.array([0])

    def count_a_lot(self):
        acc = self.acc
        self.bla = 2
        for i in range(1000000000):
            acc[0] += 1
        return acc


@jitclass(spec)
class DeepCopyClass:
    def __init__(self):
        self.bla = 0
        self.acc = np.array([0])

    def count_a_lot(self):
        acc = self.acc.copy()
        self.bla = 2
        for i in range(1000000000):
            acc[0] += 1
        return acc


@jitclass(spec)
class OutSourceClass:
    def __init__(self):
        self.acc = np.array([0])
        self.bla = 0

    def count_a_lot(self):
        return count_a_lot(self.acc)


@njit
def count_a_lot(acc):
    for i in range(1000000000):
        acc[0] += 1
    return acc


all_in_class = AllInClass()
shallow_copy_class = ShallowCopyClass()
deep_copy_class = DeepCopyClass()
out_source_class = OutSourceClass()

print("out all_in_class", all_in_class.count_a_lot())
print("out shallow_copy_class", shallow_copy_class.count_a_lot())
print("out deep_copy_class", deep_copy_class.count_a_lot())
print("out out_source_class", out_source_class.count_a_lot())
print("out pure ", count_a_lot(np.array([0])))


print("all_in_class")
s = time.time()
all_in_class.count_a_lot()
e = time.time()
print(e-s)

print("shallow_copy_class")
s = time.time()
shallow_copy_class.count_a_lot()
e = time.time()
print(e-s)

print("deep_copy_class")
s = time.time()
deep_copy_class.count_a_lot()
e = time.time()
print(e-s)

print("out_source_class")
s = time.time()
out_source_class.count_a_lot()
e = time.time()
print(e-s)

print("pure")
s = time.time()
count_a_lot(np.array([0]))
e = time.time()
print(e-s)




setup = '''
import numpy as np
from numba import njit

@njit
def numba_any(arr):
    found = False
    i = 0
    arr_len = len(arr)
    while not found and i < arr_len:
        if arr[i]:
            found = True
        i += 1
    return found



@njit
def numba_any_break(arr):
    found = False
    for i in arr:
        if i:
            found = True
            break
    return found

arr = np.array([False, False, False, False, True, False, False, False, False, False, False, False, False])
numba_any_break(arr)
numba_any(arr)

'''
import timeit

timeit.timeit('numba_any(arr)', setup=setup, number=10000000)
timeit.timeit('numba_any_break(arr)', setup=setup, number=10000000)


# arr = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False])
# numba_any(arr)
#
# setup = '''
# import numpy as np
# from numba import njit
#
# @njit
# def numba_any(arr):
#     found = False
#     i = 0
#     arr_len = len(arr)
#     while not found and i < arr_len:
#         if arr[i]:
#             found = True
#         i += 1
#     return found
#
#
#
# @njit
# def numba_any_break(arr):
#     found = False
#     for i in arr:
#         if i:
#             found = True
#             break
#     return found
#
# arr = np.array([False, False, False, False, True, False, False, False, False, False, False, False, False])
# numba_any_break(arr)
# numba_any(arr)
#
# '''
# import timeit
#
# timeit.timeit('numba_any(arr)', setup=setup, number=10000000)
# timeit.timeit('numba_any_break(arr)', setup=setup, number=10000000)
