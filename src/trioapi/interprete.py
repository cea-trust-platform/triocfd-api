import trioapi.trustify_gen as tgp

# Old add_object
# def add_object(dataset, obj, obj_identifier):
#    """
#    Add an object to the dataset.
#
#    Parameters
#    ----------
#    dataset: Dataset
#        The corresponding dataset.
#
#    obj: Objet_u
#        The corresponding object.
#
#    obj_identifier: str
#        The identifier corresponding to the object.
#    """
#    declaration = tgp.Declaration(ze_type=type(obj), identifier=obj_identifier)
#    declarations_parser = tgp.Declaration_Parser()
#    declarations_parser._tokens["cls_nam"].orig()[0] = "\n" + type(obj).__name__.lower()
#    declarations_parser._tokens["cls_nam"].low().append(type(obj).__name__.lower())
#    declaration._parser = declarations_parser
#
#    read = tgp.Read(identifier=obj_identifier, obj=obj)
#    read_parser = tgp.Read_Parser(read)
#    read_parser._tokens["cls_nam"].orig()[0] = "\nRead"
#    read_token = tgp.TRUSTTokens()
#    read_token.orig().append("\nRead")
#    read_parser._tokens.update({"read": read_token})
#    read._parser = read_parser
#    dataset.entries.insert(len(dataset.entries) - 2, declaration)
#    dataset.entries.insert(len(dataset.entries) - 2, read)
#    dataset._declarations.update(
#        {obj_identifier: [declaration, len(dataset.entries) - 3]}
#    )


def add_object(dataset, obj, obj_identifier):
    """
    Add an object to the dataset.

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    obj: Objet_u
        The corresponding object.

    obj_identifier: str
        The identifier corresponding to the object.
    """
    declaration = tgp.Declaration(ze_type=type(obj), identifier=obj_identifier)
    declarations_parser = tgp.Declaration_Parser()
    declarations_parser._tokens["cls_nam"].orig()[0] = "\n" + type(obj).__name__.lower()
    declarations_parser._tokens["cls_nam"].low().append(type(obj).__name__.lower())
    declaration._parser = declarations_parser

    read = tgp.Read(identifier=obj_identifier, obj=obj)
    read_parser = tgp.Read_Parser(read)
    read_parser._tokens["cls_nam"].orig()[0] = "\nRead"
    read_token = tgp.TRUSTTokens()
    read_token.orig().append("\nRead")
    read_parser._tokens.update({"read": read_token})
    read._parser = read_parser
    index = len(dataset.entries)
    while isinstance(dataset.entries[index - 1], (tgp.Fin, tgp.Solve)):
        index -= 1
    dataset.entries.insert(index, declaration)
    dataset.entries.insert(index + 1, read)
    dataset._declarations.update({obj_identifier: [declaration, index + 1]})


def associate_to_problem(dataset, problem_identifier, obj_identifier):
    """
    Associate an object (scheme, dom, ...) to a problem

    Parameters
    ----------
    dataset: Dataset
        The dataset in which we want to associate.

    problem_identifier: str
        The identifier corresponding to the problem.

    obj_identifier: str
        The identifier corresponding to the object.
    """
    association = tgp.Associate(objet_1=problem_identifier, objet_2=obj_identifier)
    association_parser = tgp.Associate_Parser()
    association_parser._tokens["cls_nam"].orig()[0] = "\nAssociate"
    association._parser = association_parser
    dataset.entries.insert(len(dataset.entries) - 2, association)


def solve_problem(dataset, problem_identifier):
    """
    Add the keyword python object to solve the problem.

    Parameters
    ----------
    dataset: Dataset
        The dataset in which we want to solve the problem.

    problem_identifier: str
        The identifier corresponding to the problem.
    """
    solve = tgp.Solve(pb=problem_identifier)
    solve_parser = tgp.Solve_Parser()
    solve_parser._tokens["cls_nam"].orig()[0] = "\nSolve"
    solve._parser = solve_parser
    dataset.entries.insert(len(dataset.entries) - 1, solve)


def change_dimension(dataset, new_dim):
    """
    Change the dimension of the dataset

    Parameters
    ----------
    dataset: Dataset
        The dataset in which we want to solve the problem.

    new_dim: int
        The new dimension
    """
    index = 0
    while not isinstance(dataset.entries[index], tgp.Dimension):
        index += 1
    dataset.entries[index].dim = new_dim


def add_declaration_object(dataset, obj, obj_identifier):
    """
    Add an object which is not read (only declaration) to the dataset.

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    obj: Objet_u
        The corresponding object.

    obj_identifier: str
        The identifier corresponding to the object.
    """
    declaration = tgp.Declaration(ze_type=type(obj), identifier=obj_identifier)
    declarations_parser = tgp.Declaration_Parser()
    declarations_parser._tokens["cls_nam"].orig()[0] = "\n" + type(obj).__name__.lower()
    declarations_parser._tokens["cls_nam"].low().append(type(obj).__name__.lower())
    declaration._parser = declarations_parser

    index = len(dataset.entries)
    while isinstance(dataset.entries[index - 1], (tgp.Fin, tgp.Solve)):
        index -= 1
    dataset.entries.insert(index, declaration)
    dataset._declarations.update({obj_identifier: [declaration, -1]})


def change_declaration_object(dataset, identifier, field_to_modify, new_field_value):
    """
    Change an object which is not read (only declaration) in the dataset.

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    identifier: str
        The current identifier of the object in the dataset

    field_to_modify: Literal["ze_type","identifier"]
        The field of the declaration which will be modified

    new_field_value: ?
        The new value of the field
    """
    setattr(dataset._declarations[identifier][0], field_to_modify, new_field_value)
    if field_to_modify == "identifier":
        dataset._declarations[new_field_value] = dataset._declarations.pop(identifier)
        for entry in dataset.entries:
            if isinstance(entry, tgp.Solve) and entry.pb == identifier:
                entry.pb = new_field_value
            elif isinstance(entry, tgp.Associate):
                if entry.objet_1 == identifier:
                    entry.objet_1 = new_field_value
                elif entry.objet_2 == identifier:
                    entry.objet_2 = new_field_value


def change_read_object(dataset, identifier, field_to_modify, new_field_value):
    """
    Change an object which is read in the dataset.

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    identifier: str
        The current identifier of the object in the dataset

    field_to_modify: Literal["identifier","obj"]
        The field of the object which will be modified

    new_field_value: ?
        The new value of the field
    """
    index = dataset._declarations[identifier][1]
    if field_to_modify == "obj":
        declaration_field_to_modify = "ze_type"
        declaration_new_field = type(new_field_value)
    else:
        declaration_field_to_modify = field_to_modify
        declaration_new_field = new_field_value
    change_declaration_object(
        dataset, identifier, declaration_field_to_modify, declaration_new_field
    )
    setattr(dataset.entries[index], field_to_modify, new_field_value)


def get_maillage(dataset):
    """
    Return every maillage in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    List:
        The list with every maillage object in the dataset
    """
    items_list = []
    for i in dataset.entries:
        if isinstance(i, (tgp.Mailler, tgp.Maillerparallel)):
            items_list.append(i)
    return items_list


def get_partition(dataset):
    """
    Return every partition in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    List:
        The list with every partition object in the dataset
    """
    items_list = []
    for i in dataset.entries:
        if isinstance(i, tgp.Partition):
            items_list.append(i)
    return items_list


def get_mesh(dataset):
    """
    Return every mesh in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    List:
        The list with every mesh object in the dataset
    """
    items_list = []
    for i in dataset.entries:
        if isinstance(
            i, (tgp.Read_med, tgp.Read_file, tgp.Read_file_bin, tgp.Read_tgrid)
        ):
            items_list.append(i)
    return items_list


def get_scatter(dataset):
    """
    Return every scatter in the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    List:
        The list with every scatter object in the dataset
    """
    items_list = []
    for i in dataset.entries:
        if isinstance(i, tgp.Scatter):
            items_list.append(i)
    return items_list


def get_dimension(dataset):
    """
    Return the dimension of the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    Int:
        The dimension of the dataset
    """
    index = 0
    while not isinstance(dataset.entries[index], tgp.Dimension):
        index += 1
    return dataset.entries[index].dim


def get_domain(dataset):
    """
    Return the domains of the dataset

    Parameters
    ----------
    dataset: Dataset
        The corresponding dataset.

    Returns
    -------
    List[str]:
        The names of the domains of the dataset
    """
    items_list = []
    for i, decl in dataset._declarations.items():
        if decl[0].ze_type is tgp.Domaine:
            items_list.append(i)

    return items_list


def change_type_object(initial_object, new_type):
    """
    Return the object with a new type with the common attributes between the initial one and the new one

    Parameters
    ----------
    initial_object: Pydantic object
        The corresponding object

    new_type: str
        The name of the type in which we want to transform the object

    Returns
    -------
    Pydantic object:
        The object with a new_type
    """
    new_obj = getattr(tgp, new_type)()
    common_attr = set(new_obj.__dict__) & set(type(initial_object)().__dict__)
    common_attr.remove("_parser")
    for j in common_attr:
        setattr(new_obj, j, getattr(initial_object, j))
    return new_obj
