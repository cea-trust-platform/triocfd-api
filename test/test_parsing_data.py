import pathlib
import tempfile
import trioapi

import re


def test_upwind(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = trioapi.get_jdd(str(datafile))
    sch = trioapi.get_elem(ds, trioapi.SCHEME)
    sch.tmax = 1.0
    with tempfile.TemporaryDirectory() as tmpdata:
        trioapi.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur = re.search(r"tmax(.*)", contenu)
        assert float(valeur.group(1).strip()) == 1.0
