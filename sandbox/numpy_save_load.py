from __future__ import print_function

import base64
import io
import numpy

def arr(w, h):
    return numpy.array([[i + j for j in xrange(w)] for i in xrange(h)], numpy.float)

a1 = arr(5, 10)
a2 = arr(10, 5)
print(a1)
print(a2)

output = io.BytesIO()
numpy.savez_compressed(output, **{
    '0': a1,
    '1': a2
})

bytes = output.getvalue()

output = io.BytesIO(bytes)
#output.seek(0)
data = numpy.load(output)
for i in xrange(2):
    print(i, data[str(i)])
print(data)

