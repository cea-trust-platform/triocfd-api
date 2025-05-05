import typing
import inspect


def get_attributes(cls):
    """
    Get the attributes of a specified class

    Return every attributes of the specified class in a dict format

    Parameters
    ----------
    cls: class
        The class coreesponding

    Returns
    -------
    dict:
        dictionnary representing every attributes the key is the name of the attribute and the value is the type
    """
    attr_types = {}
    if hasattr(cls, "__mro__"):
        for base in reversed(cls.__mro__):
            if hasattr(base, "__annotations__"):
                attr_types.update(base.__annotations__)
        attr_types.pop("_synonyms", None)
        return attr_types


def get_successive_attributes(cls, dict={}):
    """
    Get every attributes of class recursively

    Return every attributes of the specified class recursively

    Parameters
    ----------
    cls: class
        The class corresponding

    dict: dict
        The dict containing the current attributes to call the function recursively

    Returns
    -------
    dict:
        dictionnary representing every attributes the key is the name of the attribute and the value is his own attributes, it stops when it is classic class or class which has no attributes
    """
    iterator = iter(get_attributes(cls).items())
    while True:
        try:
            key, value = next(iterator)
            if inspect.isclass(value):
                if (
                    value not in [int, float, str, bool, list, tuple, dict, set]
                    and get_attributes(value) != {}
                ):
                    dict.update({key: get_attributes(value)})
                    get_successive_attributes(value, dict[key])
                else:
                    dict.update({key: value})
            else:
                if typing.get_origin(value) in [typing.Literal, list]:
                    dict.update({key: value})
                else:
                    if (
                        inspect.isclass(value.__args__[0])
                        and value.__args__[0]
                        not in [int, float, str, bool, list, tuple, dict, set]
                        and typing.get_origin(value.__args__[0]) is not typing.Literal
                        and get_attributes(value.__args__[0]) != {}
                    ):
                        dict.update({key: get_attributes(value.__args__[0])})
                        get_successive_attributes(value.__args__[0], dict[key])
                    else:
                        if (
                            value.__args__[0]
                            in [int, float, str, bool, list, tuple, dict, set]
                            or typing.get_origin(value.__args__[0]) is typing.Literal
                        ):
                            dict.update({key: value.__args__[0]})
                        if typing.get_origin(value.__args__[0]) is typing.Annotated:
                            dict.update(
                                {
                                    key: [
                                        get_attributes(
                                            value.__args__[0].__args__[0].__args__[0]
                                        )
                                    ]
                                }
                            )
                            get_successive_attributes(
                                value.__args__[0].__args__[0].__args__[0], dict[key][0]
                            )
                        if typing.get_origin(value) is typing.Annotated:
                            dict.update(
                                {key: [get_attributes(value.__args__[0].__args__[0])]}
                            )
                            get_successive_attributes(
                                value.__args__[0].__args__[0], dict[key][0]
                            )
                        if get_attributes(value.__args__[0]) == {}:
                            dict.update({key: value.__args__[0]})
        except StopIteration:
            break
    return dict


"""
def dict_to_object(cls,dict_attr):
    '''
    Convert a dict into the corresponding object

    Parameters
    ----------
    cls: class
        The class corresponding to the object we want to create

    dict_attr: dict
        The dictionnary which will be converted

    Returns
    -------
    obj: Pydantic class
        The object corresponding to the dictionnary
    '''
    obj=cls()
    for key,value in dict_attr.items():
        if isinstance(value,dict):
            attr_class=getattr(obj, key, None)
            if isinstance(attr_class,type):
                setattr(obj,key,dict_to_object(attr_class,value))
            else:
                if all(hasattr(attr_class,key_value) for key_value in value)and obj_to_dict(dict_to_object(type(attr_class),value)) ==value:
                    setattr(obj,key,value)
                elif attr_class is not None:
                    i=0
                    list_subclass=taj.get_subclass(type(attr_class).__name__)
                    print(list_subclass)
                    found=False
                    while i<len(list_subclass) and found==False:
                        if all(hasattr(list_subclass[i](),key_value) for key_value in value):
                            if obj_to_dict(dict_to_object(list_subclass[i],value)) == value:
                                setattr(obj,key,dict_to_object(list_subclass[i],value))
                                found=True
                        i+=1
                else :
                    setattr(obj,key,value)
        else:
            setattr(obj,key,value)
    return obj
"""


def obj_to_dict(obj):
    """
    Convert an object into the corresponding dict

    Parameters
    ----------
    obj: Pydantic class
        The object corresponding which will be converted

    Returns
    -------
    obj: dict
        The dictionnary corresponding to the object
    """

    if isinstance(obj, list):
        return [obj_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {
            k: obj_to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")
        }
    else:
        return obj


def obj_to_dict_type(obj):
    """
    Convert an object into the corresponding dict

    Parameters
    ----------
    obj: Pydantic class
        The object corresponding which will be converted

    Returns
    -------
    obj: list
        The list containing the type (1st index) and the attributes as a dict (2nd index) of the object. Every attributes which are object in the dict are also in the same format of list.
    """

    if isinstance(obj, list):
        return [obj_to_dict_type(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return [
            type(obj),
            {
                k: obj_to_dict_type(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            },
        ]
    else:
        return obj


def dict_to_object_type(list_attr):
    """
    Convert a dict into the corresponding object

    Parameters
    ----------
    list_attr: list
        A list containing the type (1st index) and the attributes as a dict (2nd index) of the object. Every object of the dict are also in the same format of list.

    Returns
    -------
    obj: Pydantic class
        The object corresponding to the list
    """
    obj = list_attr[0]()
    for key, value in list_attr[1].items():
        if (
            value is not None
            and isinstance(value, list)
            and (isinstance(value[0], list) or len(value) > 1)
        ):
            if isinstance(value[0], list):
                list_obj = []
                for i in value:
                    list_obj.append(dict_to_object_type(i))
                setattr(obj, key, list_obj)
            elif isinstance(value[1], dict):
                new_obj = dict_to_object_type(value)
                setattr(obj, key, new_obj)
            else:
                setattr(obj, key, value)
        elif value is not None:
            setattr(obj, key, value)
    return obj
