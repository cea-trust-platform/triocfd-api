import trioapi.trustify_gen as hg
from trustify.trust_parser import TRUSTParser, TRUSTStream

PROBLEM = "pb"
SCHEME = "sch"


def get_jdd(jdd_name):
    """
    Get the JDD

    Return the jdd in a python format

    Parameters
    ----------
    jdd_name: str
        The name corresponding to an existing jdd

    Returns
    -------
    Dataset:
        Pydantic class corresponding to the jdd in python format
    """
    jdd = jdd_name + ".data"
    with open(jdd) as f:
        data_ex = f.read()
    tp = TRUSTParser()
    tp.tokenize(data_ex)
    stream = TRUSTStream(tp)
    ds = hg.Dataset_Parser.ReadFromTokens(stream)
    return ds


def get_elem(dataset, type_elem):
    """
    Get the element corresponding to the object identifier in the dataset

    Parameters
    ----------
    dataset: Dataset
        The dataset in wich you are searching your object

    type_element: str
        The object identifier in the dataset

    Returns
    -------
    Pydantic class:
        Class depending on which objects you searched in the dataset
    """
    return dataset.get(type_elem)


def write_data(dataset, jdd_name):
    """
    Write the dataset in a specified data file

    Parameters
    ----------
    dataset: Dataset
        The dataset you want to write as a data file

    jdd_name: str
        The name corresponding to the data file in which the dataset will be written
    """
    jdd = jdd_name + ".data"
    newStream = dataset.toDatasetTokens()
    s = "".join(newStream)
    with open(jdd, "w") as f:
        f.write(s)
    print("Data file updated with success")
