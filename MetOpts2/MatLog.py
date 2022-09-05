def empty():
    return []


def pair(a, b):
    return [a, b]


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def sub_lists(l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j: i])
    return lists


print(sub_lists(['world', 3, 4, 'not']))


def filter(a, filt):
    res = []
    for i in a:
        if filt(i):
            res.append(i)
    return res


def f(x):
    if x > 5:
        return True
    else:
        return False


print(filter([1, 2, -4, 77, 321, 322, 321, 9, 4, 7], f))
print(list(flatten([1, 2, [3, 4, [5], ['hi']], [6, [[[7, 'hello']]]]])))
print(pair([2, 3], 1))
