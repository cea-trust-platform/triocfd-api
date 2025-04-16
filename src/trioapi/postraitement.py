import trioapi.trustify_gen as hg


def create_probe(name, special, field_name, period, type):
    """
    Create a new probe

    Parameters
    ----------
    name: str
        The name of the probe

    special: str
        Option to change the positions of the probes. Several options are available:  grav : each probe is moved to the nearest cell center of the mesh;  som : each probe is moved to the nearest vertex of the mesh  nodes : each probe is moved to the nearest face center of the mesh;  chsom : only available for P1NC sampled field. The values of the probes are calculated according to P1-Conform corresponding field.  gravcl : Extend to the domain face boundary a cell-located segment probe in order to have the boundary condition for the field. For this type the extreme probe point has to be on the face center of gravity.

    field_name: str
        Name of the sampled field

    period: float
        Period value. Every prd seconds, the field value calculated at the previous time step is written to the nom_sonde.son file

    type: Sonde_base
        Type of probe

    Returns
    -------
    Sonde:
        Pydantic class corresponding to the probe
    """
    return hg.Sonde(
        nom_sonde=name,
        special=special,
        nom_inco=field_name,
        mperiode="periode",
        prd=period,
        type=type,
    )


def create_probe_points(coos_points_list):
    """
    Define the number of probe points

    Parameters
    ----------
    coos_points_list: list
        The list of the coordinates [x,y] of each points

    Returns
    -------
    Points:
        Pydantic class corresponding to the list of points
    """

    points_list = []
    for i in coos_points_list:
        points_list.append(hg.Un_point(pos=i))
    return hg.Points(points=points_list)


def create_probe_segment(nbr_points, coos_start_point, coos_end_point):
    """
    Define the segment of probe points

    Parameters
    ----------
    nbre_points: int
        The number of points of segment

    coos_start_point: list
        The list corresponding to the coordinates of the starting point

    coos_end_point: list
        The list corresponding to the coordinates of the ending point

    Returns
    -------
    Segment:
        Pydantic class corresponding to the segment of points
    """
    return hg.Segment(
        nbr=nbr_points,
        point_deb=hg.Un_point(pos=coos_start_point),
        point_fin=hg.Un_point(pos=coos_end_point),
    )


def add_probe(problem, probe):
    """
    Add a probe to the postprocessing of the problem

    Parameters
    ----------
    problem: Pb_base
        The problem to which we want to add a probe

    probe: Sonde
        The probe we want to add to the problem

    Returns
    -------

    """
    problem.post_processing.probes.append(probe)


def add_postprocess_field(problem, field, localisation):
    """
    Add a field to be post-processed to the ist of definition champ of the problem

    Parameters
    ----------
    problem: Pb_base
        The problem we want to modify

    field: str
        The name of the post-processed field

    localisation: Literal["elem","som","faces"]
        Localisation of post-processed field values: The two available values are elem, som, or faces (LATA format only) used respectively to select field values at mesh centres (CHAMPMAILLE type field in the lml file) or at mesh nodes (CHAMPPOINT type field in the lml file). If no selection is made, localisation is set to som by default.

    Returns
    -------

    """
    new_field = hg.Champ_a_post(champ=field, localisation=localisation)
    problem.postraitement.champs.champs.append(new_field)


def get_probe_index_by_name(problem, name_probe):
    """
    Get the index of a probe in the list

    Parameters
    ----------
    problem: Pb_base
        The problem containing the postprocessing with the probe

    name_probe: str
        The name of the probe

    Return
    ------
    int:
        The index of the probe in the list of probe in the postprocessing class
    """
    return next(
        (
            i
            for i, obj in enumerate(problem.postraitement.sondes)
            if obj.nom_sonde == name_probe
        ),
        None,
    )


## TO DO ? :  def create_postprocessing()
##              definition_champs
##
##
##
##
##
##
##
##
##
