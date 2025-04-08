import pathlib

import trioapi


def test_upwind(request):
    path = pathlib.Path(request.module.__file__)
    datafile = path.parent / "upwind_simplified"
    ds = trioapi.get_jdd(str(datafile))
    print(ds)
    sch = trioapi.get_elem(ds, trioapi.SCHEME)
    sch.tmax = 1.0
    trioapi.update_data(ds, "upwind_simplified_modified")
    # TODO: utiliser re ou autre pour verifier que tmax = 1.0 strictement
