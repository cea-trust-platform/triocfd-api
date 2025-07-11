import trioapi.trustify_gen_pyd as tgp
from trioapi.jdd import get_entry_index


def delete_declaration_object(dataset, identifier):
    """
    Delete a declaration object in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    identifier: str
        The identifier corresponding to the object.
    """
    index_deleted = []
    dataset._declarations.pop(identifier)
    for i, entry in enumerate(dataset.entries):
        if (
            (isinstance(entry, tgp.Declaration) and entry.identifier == identifier)
            or (isinstance(entry, tgp.Solve) and entry.pb == identifier)
            or (
                isinstance(entry, tgp.Associate)
                and (entry.objet_1 == identifier or entry.objet_2 == identifier)
            )
            or (
                isinstance(entry, tgp.Discretize)
                and (entry.problem_name == identifier or entry.dis == identifier)
            )
        ):
            index_deleted.append(i)
    for i in sorted(index_deleted, reverse=True):
        del dataset.entries[i]
    for _, declaration in dataset._declarations.items():
        if declaration[1] != -1:
            declaration[1] -= sum(1 for i in index_deleted if i < declaration[1])


def delete_object(dataset, identifier):
    """
    Delete an object in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    identifier: str
        The identifier corresponding to the object.
    """
    index_read = dataset._declarations[identifier][1]
    del dataset.entries[index_read]
    for _, declaration in dataset._declarations.items():
        if declaration[1] != -1 and declaration[1] > index_read:
            declaration[1] -= 1
    delete_declaration_object(dataset, identifier)


def delete_read_object(dataset, obj_to_delete):
    """
    Delete a read object in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    obj_to_delete: Pydantic Object
        The object which is going to be deleted in the dataset.
    """
    index_to_delete = get_entry_index(dataset, obj_to_delete)
    del dataset.entries[index_to_delete]
    for _, declaration in dataset._declarations.items():
        if declaration[1] != -1 and declaration[1] > index_to_delete:
            declaration[1] -= 1
