import trioapi.trustify_gen as hg


def add_condlim(obj, bord, base_cl):
    condlim = hg.Condlimlu(bord=bord, cl=base_cl)
    obj.conditions_limites.append(condlim)


# def create_base_cl()
