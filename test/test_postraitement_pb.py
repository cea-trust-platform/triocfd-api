import pathlib
import tempfile
import trioapi as ta

import re


def test_probe(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = ta.get_jdd(str(datafile))
    pb = ta.get_elem(ds, ta.PROBLEM)
    segment = ta.create_probe_segment(10, [0, 1], [0, 2])
    probe = ta.create_probe("sonde_test", None, "vitesse", 0.05, segment)
    ta.add_probe(pb, probe)
    with tempfile.TemporaryDirectory() as tmpdata:
        ta.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur = re.search(
            r"sonde_vitesse vitesse periode 0.005 points 2 0.14 0.105 0.14 0.115\s*(.*?)\s*\}",
            contenu,
            re.DOTALL,
        )
        assert (
            valeur.group(1).strip()
            == "sonde_test vitesse periode 0.05 segment 10 0.0 1.0 0.0 2.0".strip()
        )


def test_postprocess_field(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = ta.get_jdd(str(datafile))
    pb = ta.get_elem(ds, ta.PROBLEM)
    ta.add_postprocess_field(pb, "pression_pa", "elem")
    with tempfile.TemporaryDirectory() as tmpdata:
        ta.write_data(ds, str(tmpdata) + "upwind_simplified_modified")
        with open(str(tmpdata) + "upwind_simplified_modified.data", "r") as f:
            contenu = f.read()
        valeur = re.search(r"vitesse elem \s*(.*?)\s*\}", contenu, re.DOTALL)
        assert valeur.group(1).strip() == "pression_pa elem".strip()
