import binascii

def read_pbm(fname):
  bytez = open("test_1.pbm", "rb").read()

  img = []
  newlines_seen = 0

  for b in bytez:
    if newlines_seen < 4:
      if b == 10:
        newlines_seen += 1
    else:
      img.append(b)

  grayscale = []

  for i in range(784):
    grayscale.append( (img[3*i] + img[3*i + 1] + img[3*i + 2]) / (3.0*255) )

  return grayscale

