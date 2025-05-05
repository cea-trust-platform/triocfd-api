import trioapi.trustify_gen as ttg
import trioapi.jdd as tj


def change_scheme(dataset, scheme_class_name):
    """
    Change the type of scheme discretization of a dataset

    Parameters
    ----------
    dataset: Dataset
        The dataset you want to change

    scheme_class_name: str
        The name corresponding to the class of the new discretization scheme that you want to implement
    """
    scheme_list = tj.get_subclass("Schema_temps_base")
    new_scheme_obj = getattr(ttg, scheme_class_name)()
    if getattr(ttg, scheme_class_name) in scheme_list:
        common_attr = set(new_scheme_obj.__dict__) & set(
            dataset._declarations["sch"][0].ze_type().__dict__
        )
        for i in dataset.entries:
            if type(i) is ttg.Declaration and i.identifier == "sch":
                i.ze_type = getattr(ttg, scheme_class_name)
                i._parser._tokens["cls_nam"]._orig = ["\n" + scheme_class_name]
                i._parser._tokens["cls_nam"]._low = [scheme_class_name.lower()]
            if type(i) is ttg.Read and i.identifier == "sch":
                for j in common_attr:
                    setattr(new_scheme_obj, j, getattr(i.obj, j))
                new_scheme_obj._parser = getattr(ttg, scheme_class_name + "_Parser")()
                i.obj = new_scheme_obj

        dataset._declarations["sch"][0].ze_type = getattr(ttg, scheme_class_name)

    else:
        print("The scheme class is not defined.")
