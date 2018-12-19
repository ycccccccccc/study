import os


dir = './cut_all'
xr = set()
xl = set()
yr = set()
yl = set()
x = set()
y = set()
for i in os.listdir(dir):
    if i.index('.')==9:
        if i[1]=='X':
            if i[0] not in xl:
                xl.add(i[0])
            if i[0] not in x:
                x.add(i[0])
            if i[2] not in xr:
                xr.add(i[2])
            if i[2] not in x:
                x.add(i[2])
        if i[1]=='Y':
            if i[0] not in yl:
                yl.add(i[0])
            if i[0] not in y:
                y.add(i[0])
            if i[2] not in yr:
                yr.add(i[2])
            if i[2] not in y:
                y.add(i[2])
print('x左：',sorted(xl))
print('x右：',sorted(xr))
print('y左：',sorted(yl))
print('y右：',sorted(yr))
print('x：',sorted(x))
print('y：',sorted(y))