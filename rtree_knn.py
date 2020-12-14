import pickle
import time
from rtree import index
from Secuential_KNN.py import read_Face

QUERY = read_Face('./unknownFaces/jennifer_lopez.jpg')

def parse_row(cara):
    l = cara
    nl = [x for pair in zip(l, l) for x in pair]
    return tuple(nl)

def load_n(i):
    nl = {}
    with open("result.pkl", "rb") as f:
        data = pickle.load(f)
        contador = 0 
        for d in data:
            if (contador == i):
                break
            nl[d[0]] = parse_row(d[1])
            contador += 1

    prop = index.Property()
    prop.dat_extension = "data"
    prop.idx_extension = "index"
    prop.dimension = 128
    idx = index.Index('3d_index' + str(i), properties = prop, interleaved = False)
    c_id = 0
    for key in nl:
        idx.insert(c_id, nl[key], key)
        c_id+=1
    return idx


list_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800] 
for i in list_sizes:
    idx = load_n(i)
    start_time = time.time()
    res = list(idx.nearest(parse_row(QUERY), 16))
    print("--- %s seconds ---" % (time.time() - start_time))
