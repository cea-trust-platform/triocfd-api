import trioapi.trustify_gen as tgp


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
    dataset.entries.insert(len(dataset.entries) - 2, declaration)
    dataset.entries.insert(len(dataset.entries) - 2, read)
    dataset._declarations.update(
        {obj_identifier: [declaration, len(dataset.entries) - 3]}
    )


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
