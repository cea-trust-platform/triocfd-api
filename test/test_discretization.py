import pathlib
import tempfile
import trioapi as ta

import re


def test_probe(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = ta.get_jdd(str(datafile))
    ta.change_scheme(ds, "Schema_implicite_base")
    with tempfile.TemporaryDirectory() as tmpdata:
        ta.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur = re.search(
            r"dis(.*?)sch",
            contenu,
            re.DOTALL,
        )
        assert valeur.group(1).strip() == "Schema_implicite_base"
