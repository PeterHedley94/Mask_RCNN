def outer_it(x, y, out=None):
    mulop = np.multiply

    it = np.nditer([x, y, out], ['external_loop'],
            [['readonly'], ['readonly'], ['writeonly', 'allocate']],
            op_axes=[range(x.ndim)+[-1]*y.ndim,
                     [-1]*x.ndim+range(y.ndim),
                     None])

    for (a, b, c) in it:
        mulop(a, b, out=c)

    return it.operands[2]


a = np.arange(2)+1
b = np.arange(3)+1

print(a)
print(b)

c = outer_it(a,b)

print(c)
