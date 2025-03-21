import sys

fname = sys.argv[-1]

for line in open(fname).readlines()[1:]:
    u = line.split(' ')[0]
    v = line.split(' ')[1].rstrip('\n')

    print(",".join([u,v]))
