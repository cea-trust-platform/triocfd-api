import trioapi.trustify_gen as hg
from trustify.trust_parser import TRUSTParser, TRUSTStream
import trioapi.trustify_gen_pyd as tgp
import inspect
from pathlib import Path
import importlib.resources as resources
import os

PROBLEM = "pb"
SCHEME = "sch"


def get_jdd(jdd_name, repository="trioapi.data"):
    """
    Get the JDD

    Return the jdd in a python format

    Parameters
    ----------
    jdd_name: str
        The name corresponding to an existing jdd

    repository: str
        The name of the repository

    Returns
    -------
    Dataset:
        Pydantic class corresponding to the jdd in python format
    """
    jdd_filename = f"{jdd_name}.data"

    try:
        # Verify qi c'est un dossier valide
        repo_path = Path(repository)
        if repo_path.exists() and repo_path.is_dir():
            jdd_path = repo_path / jdd_filename
            with open(jdd_path, "r") as f:
                data_ex = f.read()
        else:
            # Si c'est le package trioapi.data
            with resources.files(repository).joinpath(jdd_filename).open("r") as f:
                data_ex = f.read()
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load JDD file '{jdd_filename}' from '{repository}': {e}"
        )
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


def write_data(dataset, jdd_name, repository=None):
    """
    Write the dataset in a specified data file

    Parameters
    ----------
    dataset: Dataset
        The dataset you want to write as a data file

    jdd_name: str
        The name corresponding to the data file in which the dataset will be written

    repository: str
        The name of the repository
    """
    jdd = f"{jdd_name}.data"
    newStream = dataset.toDatasetTokens()
    s = "".join(newStream)

    if repository is not None:
        complete_path = os.path.join(repository, jdd)
        with open(complete_path, "w") as f:
            f.write(s)
    else:
        with open(jdd, "w") as f:
            f.write(s)
    print("Data file created with success.")


def get_subclass(class_name):
    """
    Get all the subclass of a specified trustify class

    Parameters
    ----------
    class_name :str
        The name of the class

    Returns
    -------
    List:
        List of every class that are subclass
    """
    return [
        cls
        for nom, cls in inspect.getmembers(tgp, inspect.isclass)
        if issubclass(cls, getattr(tgp, class_name)) and cls != getattr(tgp, class_name)
    ]


def get_read_objects(dataset):
    """
    Get all the object identifier of the dataset which have been Read in the data file

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset

    Returns
    -------
    List:
        List of every object identifier that have been Read
    """
    items_list = []
    for key, value in dataset._declarations.items():
        if value[1] > 0:
            items_list.append(key)
    return items_list


def get_read_pb(dataset):
    """
    Get all the pb of the dataset which have been Read in the data file

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset

    Returns
    -------
    List:
        List of tuple (identifier, type) of every problem of the dataset
    """
    items_list = []
    for key, value in dataset._declarations.items():
        if value[1] > 0 and value[0].ze_type in get_subclass(tgp.Pb_base.__name__):
            items_list.append([key, dataset.get(key)])
    return items_list


def get_read_sch(dataset):
    """
    Get all the scheme of the dataset which have been Read in the data file

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset

    Returns
    -------
    List:
        List of tuple (identifier, type) of every scheme of the dataset
    """
    items_list = []
    for key, value in dataset._declarations.items():
        if value[1] > 0 and value[0].ze_type in get_subclass(
            tgp.Schema_temps_base.__name__
        ):
            items_list.append([key, dataset.get(key)])
    return items_list


def get_dis(dataset):
    items_list = []
    for key, value in dataset._declarations.items():
        if value[0].ze_type in get_subclass("Discretisation_base"):
            items_list.append([key, value[0].ze_type])
    return items_list


def get_description_as_dict(type_object):
    """
    Get all the description of the attributes of the type as a dict format

    Parameters
    ----------
    type_object: class
        The corresponding object

    Returns
    -------
    Dict:
        Dict of every description for each attributes
    """
    dict_description = {}
    if type_object is not None:
        for name, field in type_object.model_fields.items():
            if field.description:
                desc = field.description.replace("\n", " ").strip()
                dict_description.update({str(name): str(desc)})
    return dict_description


def get_entry_index(dataset, created_object):
    """
    Get the index of an object in the dataset entries

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset

    created_object: Pydantic object
        The actual object for which we want to know the index in the dataset entries

    Returns
    -------
    Int:
        Index of the object in the entries of the dataset
    """

    index = 0
    while index < len(dataset.entries) and dataset.entries[index] != created_object:
        index += 1
    if index == len(dataset.entries):
        return -1
    else:
        return index
