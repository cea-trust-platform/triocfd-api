import pathlib
import tempfile
import trioapi as ta

import re


def test_add_solve_pb(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = ta.get_jdd(str(datafile))
    ta.add_object(ds, ta.trustify_gen_pyd.Pb_hydraulique(), "pb2")
    ta.solve_problem(ds, "pb2")
    with tempfile.TemporaryDirectory() as tmpdata:
        ta.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur1 = re.findall(
            r"Read (.*)",
            contenu,
        )
        valeur2 = re.findall(
            r"Solve (.*)",
            contenu,
        )
        assert "pb2 {" in valeur1 and "pb2" in valeur2


def test_associate(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = ta.get_jdd(str(datafile))
    ta.add_object(ds, ta.trustify_gen_pyd.Pb_hydraulique(), "pb2")
    ta.associate_to_problem(ds, "pb2", "sch")
    with tempfile.TemporaryDirectory() as tmpdata:
        ta.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur = re.search(
            r"Associate pb2(.*)",
            contenu,
        )
        assert valeur.group(1).strip() == "sch"
