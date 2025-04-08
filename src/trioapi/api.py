import trustify.trustify_gen as hg
from trustify.trust_parser import TRUSTParser, TRUSTStream

PROBLEM = "pb"
SCHEME = "sch"


def get_jdd(cle_jdd):
    jdd = cle_jdd + ".data"
    with open(jdd) as f:
        data_ex = f.read()
    tp = TRUSTParser()
    tp.tokenize(data_ex)
    stream = TRUSTStream(tp)
    ds = hg.Dataset_Parser.ReadFromTokens(stream)
    return ds


def get_elem(dataset, type_elem):
    return dataset.get(type_elem)


def update_data(dataset, cle_jdd):
    jdd = cle_jdd + ".data"
    newStream = dataset.toDatasetTokens()
    s = "".join(newStream)
    with open(jdd, "w") as f:
        f.write(s)
    print("Data file updated with success")
