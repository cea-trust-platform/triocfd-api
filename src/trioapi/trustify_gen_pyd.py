################################################################
# This file was generated automatically from :
# /volatile/catB/tm283032/triocfd/build/trustify/generated/TRAD2_trustify
# 25-03-27 at 09:31:28
################################################################


from typing import Annotated, ClassVar, List, Literal, Optional, Any, Dict
import pydantic
from pydantic import ConfigDict, Field, create_model


class TRUSTBaseModel(pydantic.BaseModel):
    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())

    @classmethod
    def with_fields(cls, **field_definitions):
        """Overriding this classmethod allows to dynamically create new pydantic models
        with fields added dynamically. This is used in hacks.py for FT problems ...
        See https://github.com/pydantic/pydantic/issues/1937
        """
        return create_model(cls.__name__, __base__=cls, **field_definitions)

    def __init__(self, *args, **kwargs):
        pydantic.BaseModel.__init__(self, *args, **kwargs)
        self._parser = None  #: An instance of AbstractCommon_Parser that is used to build the current pydantic object

    def self_validate(self):
        """Validate an instance to see if it complies with the pydantic schema"""
        dmp = self.__class__.model_dump(self, warnings=False, serialize_as_any=True)
        self.__class__.model_validate(dmp)

    def toDatasetTokens(self):
        """Convert a pydantic object (self) back to a stream of tokens that can be output in a file to reproduce
        a TRUST dataset."""
        from trustify.misc_utilities import ClassFactory

        if self._parser is None:
            self._parser = ClassFactory.GetParserFromPyd(self.__class__)()
        self._parser._pyd_value = self
        return self._parser.toDatasetTokens()

    def __getattribute__(self, nam):
        """Override to allow the (scripting) user to use attribute synonyms, for
        example 'pb.post_processing' instead of 'pb.postraitement' ...
        """
        cls = super().__getattribute__("__class__")
        if nam not in super().__getattribute__("model_fields"):
            # Not very efficient, but the lists should never be too big ...
            while cls is not TRUSTBaseModel:
                for attr_nam, lst_syno in cls._synonyms.items():
                    if attr_nam is not None and nam in lst_syno:
                        return super().__getattribute__(attr_nam)
                cls = cls.__base__  # Go one level up
        return super().__getattribute__(nam)

    def __setattr__(self, nam, val):
        """Override to allow the (scripting) user to use attribute synonyms, for
        example 'pb.post_processing' instead of 'pb.postraitement' ...
        """
        # Go over all base class to try to find a synonym
        cls = self.__class__
        if nam not in self.model_fields:
            # Not very efficient, but the lists should never be too big ...
            while cls is not TRUSTBaseModel:
                for attr_nam, lst_syno in cls._synonyms.items():
                    if attr_nam is not None and nam in lst_syno:
                        return super().__setattr__(nam, val)
                cls = cls.__base__  # Go one level up
        return super().__setattr__(nam, val)


################################################################


class Objet_u(TRUSTBaseModel):
    r""" """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Comment(Objet_u):
    r"""
    Comments in a data file.
    """

    comm: str = Field(description=r"""Text to be commented.""", default="")
    _synonyms: ClassVar[dict] = {None: ["#"], "comm": []}


################################################################


class Bloc_comment(Objet_u):
    r"""
    bloc of Comment in a data file.
    """

    comm: str = Field(description=r"""Text to be commented.""", default="")
    _synonyms: ClassVar[dict] = {None: ["/*"], "comm": []}


################################################################


class Listobj_impl(TRUSTBaseModel):
    r""" """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Listobj(Listobj_impl):
    r"""
    List of objects.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Objet_lecture(Objet_u):
    r"""
    Auxiliary class for reading.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Bloc_lecture(Objet_lecture):
    r"""
    to read between two braces
    """

    bloc_lecture: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "bloc_lecture": []}


################################################################


class Deuxmots(Objet_lecture):
    r"""
    Two words.
    """

    mot_1: str = Field(description=r"""First word.""", default="")
    mot_2: str = Field(description=r"""Second word.""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot_1": [], "mot_2": []}


################################################################


class Troismots(Objet_lecture):
    r"""
    Three words.
    """

    mot_1: str = Field(description=r"""First word.""", default="")
    mot_2: str = Field(description=r"""Snd word.""", default="")
    mot_3: str = Field(description=r"""Third word.""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot_1": [], "mot_2": [], "mot_3": []}


################################################################


class Format_file(Objet_lecture):
    r"""
    File formatted.
    """

    format: Optional[Literal["binaire", "formatte", "xyz", "single_hdf"]] = Field(
        description=r"""Type of file (the file format).""", default=None
    )
    name_file: str = Field(description=r"""Name of file.""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "format": [], "name_file": []}


################################################################


class Deuxentiers(Objet_lecture):
    r"""
    Two integers.
    """

    int1: int = Field(description=r"""First integer.""", default=0)
    int2: int = Field(description=r"""Second integer.""", default=0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "int1": [], "int2": []}


################################################################


class Floatfloat(Objet_lecture):
    r"""
    Two reals.
    """

    a: float = Field(description=r"""First real.""", default=0.0)
    b: float = Field(description=r"""Second real.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "a": [], "b": []}


################################################################


class Entierfloat(Objet_lecture):
    r"""
    An integer and a real.
    """

    the_int: int = Field(description=r"""Integer.""", default=0)
    the_float: float = Field(description=r"""Real.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "the_int": [], "the_float": []}


################################################################


class Class_generic(Objet_u):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Listchamp_generique(Listobj):
    r"""
    XXX
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class List_list_nom(Listobj):
    r"""
    pour les groupes
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Source_base(Objet_u):
    r"""
    Basic class of source terms introduced in the equation.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Darcy(Source_base):
    r"""
    Class for calculation in a porous media with source term of Darcy -nu/K*V. This keyword
    must be used with a permeability model. For the moment there are two models : permeability
    constant or Ergun\'s law. Darcy source term is available for quasi compressible
    calculation. A new keyword is aded for porosity (porosite).
    """

    bloc: Bloc_lecture = Field(
        description=r"""Description.""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Interprete(Objet_u):
    r"""
    Basic class for interpreting a data file. Interpretors allow some operations to be carried
    out on objects.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Debut_bloc(Interprete):
    r"""
    Block\'s beginning.
    """

    _synonyms: ClassVar[dict] = {None: ["{"]}


################################################################


class Fin_bloc(Interprete):
    r"""
    Block\'s end.
    """

    _synonyms: ClassVar[dict] = {None: ["}"]}


################################################################


class Export(Interprete):
    r"""
    Class to make the object have a global range, if not its range will apply to the block
    only (the associated object will be destroyed on exiting the block).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Troisf(Objet_lecture):
    r"""
    Auxiliary class to extrude.
    """

    lx: float = Field(
        description=r"""X direction of the extrude operation.""", default=0.0
    )
    ly: float = Field(
        description=r"""Y direction of the extrude operation.""", default=0.0
    )
    lz: float = Field(
        description=r"""Z direction of the extrude operation.""", default=0.0
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "lx": [], "ly": [], "lz": []}


################################################################


class Floatentier(Objet_lecture):
    r"""
    A real and an integer.
    """

    the_float: float = Field(description=r"""Real.""", default=0.0)
    the_int: int = Field(description=r"""Integer.""", default=0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "the_float": [], "the_int": []}


################################################################


class Turbulence_paroi_base(Objet_u):
    r"""
    Basic class for wall laws for Navier-Stokes equations.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Dt_impr_ustar_mean_only(Objet_lecture):
    r"""
    not_set
    """

    dt_impr: float = Field(description=r"""not_set""", default=0.0)
    boundaries: Optional[List[str]] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["nul"], "dt_impr": [], "boundaries": []}


################################################################


class Modele_turbulence_hyd_deriv(Objet_lecture):
    r"""
    Basic class for turbulence model for Navier-Stokes equations.
    """

    turbulence_paroi: Optional[Turbulence_paroi_base] = Field(
        description=r"""Keyword to set the wall law.""", default=None
    )
    dt_impr_ustar: Optional[float] = Field(
        description=r"""This keyword is used to print the values (U +, d+, u$\star$) obtained with the wall laws into a file named datafile_ProblemName_Ustar.face and periode refers to the printing period, this value is expressed in seconds.""",
        default=None,
    )
    dt_impr_ustar_mean_only: Optional[Dt_impr_ustar_mean_only] = Field(
        description=r"""This keyword is used to print the mean values of u* ( obtained with the wall laws) on each boundary, into a file named datafile_ProblemName_Ustar_mean_only.out. periode refers to the printing period, this value is expressed in seconds. If you don\'t use the optional keyword boundaries, all the boundaries will be considered. If you use it, you must specify nb_boundaries which is the number of boundaries on which you want to calculate the mean values of u*, then you have to specify their names.""",
        default=None,
    )
    nut_max: Optional[float] = Field(
        description=r"""Upper limitation of turbulent viscosity (default value 1.e8).""",
        default=None,
    )
    correction_visco_turb_pour_controle_pas_de_temps: Optional[bool] = Field(
        description=r"""Keyword to set a limitation to low time steps due to high values of turbulent viscosity. The limit for turbulent viscosity is calculated so that diffusive time-step is equal or higher than convective time-step. For a stationary flow, the correction for turbulent viscosity should apply only during the first time steps and not when permanent state is reached. To check that, we could post process the corr_visco_turb field which is the correction of turbulent viscosity: it should be 1. on the whole domain.""",
        default=None,
    )
    correction_visco_turb_pour_controle_pas_de_temps_parametre: Optional[float] = Field(
        description=r"""Keyword to set a limitation to low time steps due to high values of turbulent viscosity. The limit for turbulent viscosity is the ratio between diffusive time-step and convective time-step is higher or equal to the given value [0-1]""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Form_a_nb_points(Objet_lecture):
    r"""
    The structure fonction is calculated on nb points and we should add the 2 directions
    (0:OX, 1:OY, 2:OZ) constituting the homegeneity planes. Example for channel flows, planes
    parallel to the walls.
    """

    nb: Literal[4] = Field(description=r"""Number of points.""", default=4)
    dir1: int = Field(description=r"""First direction.""", default=0, le=2)
    dir2: int = Field(description=r"""Second direction.""", default=0, le=2)
    _synonyms: ClassVar[dict] = {None: ["nul"], "nb": [], "dir1": [], "dir2": []}


################################################################


class Mod_turb_hyd_ss_maille(Modele_turbulence_hyd_deriv):
    r"""
    Class for sub-grid turbulence model for Navier-Stokes equations.
    """

    formulation_a_nb_points: Optional[Form_a_nb_points] = Field(
        description=r"""The structure fonction is calculated on nb points and we should add the 2 directions (0:OX, 1:OY, 2:OZ) constituting the homegeneity planes. Example for channel flows, planes parallel to the walls.""",
        default=None,
    )
    longueur_maille: Optional[
        Literal["volume", "volume_sans_lissage", "scotti", "arrete"]
    ] = Field(
        description=r"""Different ways to calculate the characteristic length may be specified :  volume : It is the default option. Characteristic length is based on the cubic root of the volume cells. A smoothing procedure is applied to avoid discontinuities of this quantity in VEF from a cell to another.  volume_sans_lissage : For VEF only. Characteristic length is based on the cubic root of the volume cells (without smoothing procedure). scotti : Characteristic length is based on the cubic root of the volume cells and the Scotti correction is applied to take into account the stretching of the cell in the case of anisotropic meshes.  arete : For VEF only. Characteristic length relies on the max edge (+ smoothing procedure) is taken into account.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_selectif_mod(Mod_turb_hyd_ss_maille):
    r"""
    Selective structure sub-grid function model (modified).
    """

    thi: Optional[Deuxentiers] = Field(
        description=r"""For homogeneous isotropic turbulence (THI), two integers ki and kc are needed in VDF (not in VEF).""",
        default=None,
    )
    canal: Optional[Floatentier] = Field(
        description=r"""h dir_faces_paroi: For a channel flow, the half width h and the orientation of the wall dir_faces_paroi are needed.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "thi": [],
        "canal": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_selectif(Mod_turb_hyd_ss_maille):
    r"""
    Selective structure sub-grid function model (a filter is applied to the structure
    function).
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_1elt(Mod_turb_hyd_ss_maille):
    r"""
    Turbulence model sous_maille_1elt.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_1elt_selectif_mod(Sous_maille_1elt):
    r"""
    Turbulence model sous_maille_1elt_selectif_mod.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_axi(Mod_turb_hyd_ss_maille):
    r"""
    Structure sub-grid function turbulence model available in cylindrical co-ordinates.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_smago_filtre(Mod_turb_hyd_ss_maille):
    r"""
    Smagorinsky sub-grid turbulence model should be used with low-filter.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_smago_dyn(Mod_turb_hyd_ss_maille):
    r"""
    Dynamic Smagorinsky sub-grid turbulence model (available in VDF discretization only).
    """

    stabilise: Optional[Literal["6_points", "moy_euler", "plans_paralleles"]] = Field(
        description=r"""not_set""", default=None
    )
    nb_points: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "stabilise": [],
        "nb_points": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Dt_impr_nusselt_mean_only(Objet_lecture):
    r"""
    not_set
    """

    dt_impr: float = Field(description=r"""not_set""", default=0.0)
    boundaries: Optional[List[str]] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["nul"], "dt_impr": [], "boundaries": []}


################################################################


class Turbulence_paroi_scalaire_base(Objet_u):
    r"""
    Basic class for wall laws for energy equation.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Modele_turbulence_scal_base(Objet_u):
    r"""
    Basic class for turbulence model for energy equation.
    """

    dt_impr_nusselt: Optional[float] = Field(
        description=r"""Keyword to print local values of Nusselt number and temperature near a wall during a turbulent calculation. The values will be printed in the _Nusselt.face file each dt_impr_nusselt time period. The local Nusselt expression is as follows : Nu = ((lambda+lambda_t)/lambda)*d_wall/d_eq where d_wall is the distance from the first mesh to the wall and d_eq is given by the wall law. This option also gives the value of d_eq and h = (lambda+lambda_t)/d_eq and the fluid temperature of the first mesh near the wall.  For the Neumann boundary conditions (flux_impose), the <<equivalent>> wall temperature given by the wall law is also printed (Tparoi equiv.) preceded for VEF calculation by the edge temperature <<T face de bord>>.""",
        default=None,
    )
    dt_impr_nusselt_mean_only: Optional[Dt_impr_nusselt_mean_only] = Field(
        description=r"""This keyword is used to print the mean values of Nusselt ( obtained with the wall laws) on each boundary, into a file named datafile_ProblemName_nusselt_mean_only.out. periode refers to the printing period, this value is expressed in seconds. If you don\'t use the optional keyword boundaries, all the boundaries will be considered. If you use it, you must specify nb_boundaries which is the number of boundaries on which you want to calculate the mean values, then you have to specify their names.""",
        default=None,
    )
    turbulence_paroi: Optional[Turbulence_paroi_scalaire_base] = Field(
        description=r"""Keyword to set the wall law.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dt_impr_nusselt": [],
        "dt_impr_nusselt_mean_only": [],
        "turbulence_paroi": [],
    }


################################################################


class Sous_maille_dyn(Modele_turbulence_scal_base):
    r"""
    Dynamic sub-grid turbulence modele.

    Warning : Available in VDF only. Not coded in VEF yet.
    """

    stabilise: Optional[Literal["6_points", "moy_euler", "plans_paralleles"]] = Field(
        description=r"""not_set""", default=None
    )
    nb_points: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "stabilise": [],
        "nb_points": [],
        "dt_impr_nusselt": [],
        "dt_impr_nusselt_mean_only": [],
        "turbulence_paroi": [],
    }


################################################################


class Loi_odvm(Turbulence_paroi_scalaire_base):
    r"""
    Thermal wall-function based on the simultaneous 1D resolution of a turbulent thermal
    boundary-layer and a variance transport equation, adapted to conjugate heat-transfer
    problems with fluid/solid thermal interaction (where a specific boundary condition should
    be used : Paroi_Echange_Contact_OVDM_VDF). This law is also available with isothermal
    walls.
    """

    n: int = Field(
        description=r"""Number of points per face in the 1D uniform meshes. n should be choosen in order to have the first point situated near $\Delta$ y+=1/3.""",
        default=0,
    )
    gamma: float = Field(
        description=r"""Smoothing parameter of the signal between 10e-5 (no smoothing) and 10e-1 (high averaging).""",
        default=0.0,
    )
    stats: Optional[Floatfloat] = Field(
        description=r"""value_t0 value_dt : Only for plane channel flow, it gives mean and root mean square profiles in the fine meshes, since value_t0 and every value_dt seconds. The values are printed into files named ODVM_fields*.dat.""",
        default=None,
    )
    check_files: Optional[bool] = Field(
        description=r"""It gives for one boundary face a historical view of local instantaneous and filtered values, as well as the calculated variance profiles from the resolution of the equation. The printed values are into the file Suivi_ndeb.dat.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "n": [],
        "gamma": [],
        "stats": [],
        "check_files": [],
    }


################################################################


class Loi_ww_scalaire(Turbulence_paroi_scalaire_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_puissance_hydr(Turbulence_paroi_base):
    r"""
    A Loi_puissance_hydr law for wall turbulence for NAVIER STOKES equations.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_standard_hydr(Turbulence_paroi_base):
    r"""
    Keyword for the logarithmic wall law for a hydraulic problem. Loi_standard_hydr refers to
    first cell rank eddy-viscosity defined from continuous analytical functions, whereas
    Loi_standard_hydr_3couches from functions separataly defined for each sub-layer
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_ww_hydr(Loi_standard_hydr):
    r"""
    laws have been qualified on channel calculation
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_ciofalo_hydr(Loi_standard_hydr):
    r"""
    A Loi_ciofalo_hydr law for wall turbulence for NAVIER STOKES equations.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Condlim_base(Objet_u):
    r"""
    Basic class of boundary conditions.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_echange_contact_vdf(Condlim_base):
    r"""
    Boundary condition type to model the heat flux between two problems. Important: the name
    of the boundaries in the two problems should be the same.
    """

    autrepb: str = Field(description=r"""Name of other problem.""", default="")
    nameb: str = Field(description=r"""Name of bord.""", default="")
    temp: str = Field(description=r"""Name of field.""", default="")
    h: float = Field(
        description=r"""Value assigned to a coefficient (expressed in W.K-1m-2) that characterises the contact between the two mediums. In order to model perfect contact, h must be taken to be infinite. This value must obviously be the same in both the two problems blocks.  The surface thermal flux exchanged between the two mediums is represented by :  fi = h (T1-T2) where 1/h = d1/lambda1 + 1/val_h_contact + d2/lambda2  where di : distance between the node where Ti and the wall is found.""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "temp": [],
        "h": [],
    }


################################################################


class Paroi_echange_contact_odvm_vdf(Paroi_echange_contact_vdf):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "temp": [],
        "h": [],
    }


################################################################


class Front_field_base(Objet_u):
    r"""
    Basic class for fields at domain boundaries.
    """

    _synonyms: ClassVar[dict] = {None: ["champ_front_base"]}


################################################################


class Champ_front_vortex(Front_field_base):
    r"""
    not_set
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    geom: str = Field(description=r"""not_set""", default="")
    nu: float = Field(description=r"""not_set""", default=0.0)
    utau: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "geom": [], "nu": [], "utau": []}


################################################################


class Traitement_particulier_base(Objet_lecture):
    r"""
    Basic class to post-process particular values.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Thi(Traitement_particulier_base):
    r"""
    Keyword for a THI (Homogeneous Isotropic Turbulence) calculation.
    """

    init_ec: int = Field(
        description=r"""Keyword to renormalize initial velocity so that kinetic energy equals to the value given by keyword val_Ec.""",
        default=0,
    )
    val_ec: Optional[float] = Field(
        description=r"""Keyword to impose a value for kinetic energy by velocity renormalizated if init_Ec value is 1.""",
        default=None,
    )
    facon_init: Optional[Literal[0, 1]] = Field(
        description=r"""Keyword to specify how kinetic energy is computed (0 or 1).""",
        default=None,
    )
    calc_spectre: Optional[Literal[0, 1]] = Field(
        description=r"""Calculate or not the spectrum of kinetic energy.  Files called Sorties_THI are written with inside four columns :  time:t global_kinetic_energy:Ec enstrophy:D skewness:S  If calc_spectre is set to 1, a file Sorties_THI2_2 is written with three columns :  time:t kinetic_energy_at_kc=32 enstrophy_at_kc=32  If calc_spectre is set to 1, a file spectre_xxxxx is written with two columns at each time xxxxx :  frequency:k energy:E(k).""",
        default=None,
    )
    periode_calc_spectre: Optional[float] = Field(
        description=r"""Period for calculating spectrum of kinetic energy""",
        default=None,
    )
    spectre_3d: Optional[Literal[0, 1]] = Field(
        description=r"""Calculate or not the 3D spectrum""", default=None
    )
    spectre_1d: Optional[Literal[0, 1]] = Field(
        description=r"""Calculate or not the 1D spectrum""", default=None
    )
    conservation_ec: Optional[bool] = Field(
        description=r"""If set to 1, velocity field will be changed as to have a constant kinetic energy (default 0)""",
        default=None,
    )
    longueur_boite: Optional[float] = Field(
        description=r"""Length of the calculation domain""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "init_ec": [],
        "val_ec": [],
        "facon_init": [],
        "calc_spectre": [],
        "periode_calc_spectre": [],
        "spectre_3d": [],
        "spectre_1d": [],
        "conservation_ec": [],
        "longueur_boite": [],
    }


################################################################


class Thi_thermo(Thi):
    r"""
    Treatment for the temperature field.

    It offers the possibility to :

    - evaluate the probability density function on temperature field,

    - give in a file the temperature field for a future spectral analysis,

    - monitor the evolution of the max and min temperature on the whole domain.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "init_ec": [],
        "val_ec": [],
        "facon_init": [],
        "calc_spectre": [],
        "periode_calc_spectre": [],
        "spectre_3d": [],
        "spectre_1d": [],
        "conservation_ec": [],
        "longueur_boite": [],
    }


################################################################


class Brech(Traitement_particulier_base):
    r"""
    non documente
    """

    bloc: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Combinaison(Mod_turb_hyd_ss_maille):
    r"""
    This keyword specifies a turbulent viscosity model where the turbulent viscosity is user-
    defined.
    """

    nb_var: Optional[List[str]] = Field(
        description=r"""Number and names of variables which will be used in the turbulent viscosity definition (by default 0)""",
        default=None,
    )
    fonction: Optional[str] = Field(
        description=r"""Fonction for turbulent viscosity. X,Y,Z and variables defined previously can be used.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nb_var": [],
        "fonction": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille(Mod_turb_hyd_ss_maille):
    r"""
    Structure sub-grid function model.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Modele_fonction_bas_reynolds_base(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Mor_eqn(Objet_u):
    r"""
    Class of equation pieces (morceaux d\'equation).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Convection_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Bloc_convection(Objet_lecture):
    r"""
    not_set
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    operateur: Convection_deriv = Field(
        description=r"""not_set""", default_factory=lambda: eval("Convection_deriv()")
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {None: ["nul"], "aco": [], "operateur": [], "acof": []}


################################################################


class Diffusion_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Solveur_sys_base(Class_generic):
    r"""
    Basic class to solve the linear system.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Op_implicite(Objet_lecture):
    r"""
    not_set
    """

    implicite: Literal["implicite"] = Field(
        description=r"""not_set""", default="implicite"
    )
    mot: Literal["solveur"] = Field(description=r"""not_set""", default="solveur")
    solveur: Solveur_sys_base = Field(
        description=r"""not_set""", default_factory=lambda: eval("Solveur_sys_base()")
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "implicite": [],
        "mot": [],
        "solveur": [],
    }


################################################################


class Bloc_diffusion(Objet_lecture):
    r"""
    not_set
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    operateur: Optional[Diffusion_deriv] = Field(
        description=r"""if none is specified, the diffusive scheme used is a 2nd-order scheme.""",
        default=None,
    )
    op_implicite: Optional[Op_implicite] = Field(
        description=r"""To have diffusive implicitation, it use Uzawa algorithm. Very useful when viscosity has large variations.""",
        default=None,
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "aco": [],
        "operateur": [],
        "op_implicite": [],
        "acof": [],
    }


################################################################


class Condlims(Listobj):
    r"""
    Boundary conditions.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Condinits(Listobj):
    r"""
    Initial conditions.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Sources(Listobj):
    r"""
    The sources.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Ecrire_fichier_xyz_valeur(Interprete):
    r"""
    This keyword is used to write the values of a field only for some boundaries in a text
    file with the following format: n_valeur

    x_1 y_1 [z_1] val_1

    ...

    x_n y_n [z_n] val_n

    The created files are named : pbname_fieldname_[boundaryname]_time.dat
    """

    binary_file: Optional[bool] = Field(
        description=r"""To write file in binary format""", default=None
    )
    dt: Optional[float] = Field(description=r"""File writing frequency""", default=None)
    fields: Optional[List[str]] = Field(
        description=r"""Names of the fields we want to write""", default=None
    )
    boundaries: Optional[List[str]] = Field(
        description=r"""Names of the boundaries on which to write fields""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "binary_file": [],
        "dt": [],
        "fields": [],
        "boundaries": [],
    }


################################################################


class Parametre_equation_base(Objet_lecture):
    r"""
    Basic class for parametre_equation
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Condlimlu(Objet_lecture):
    r"""
    Boundary condition specified.
    """

    bord: str = Field(
        description=r"""Name of the edge where the boundary condition applies.""",
        default="",
    )
    cl: Condlim_base = Field(
        description=r"""Boundary condition at the boundary called bord (edge).""",
        default_factory=lambda: eval("Condlim_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "bord": [], "cl": []}


################################################################


class Field_base(Objet_u):
    r"""
    Basic class of fields.
    """

    _synonyms: ClassVar[dict] = {None: ["champ_base"]}


################################################################


class Condinit(Objet_lecture):
    r"""
    Initial condition.
    """

    nom: str = Field(description=r"""Name of initial condition field.""", default="")
    ch: Field_base = Field(
        description=r"""Type field and the initial values.""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "nom": [], "ch": []}


################################################################


class Eqn_base(Mor_eqn):
    r"""
    Basic class for equations.
    """

    disable_equation_residual: Optional[str] = Field(
        description=r"""The equation residual will not be used for the problem residual used when checking time convergence or computing dynamic time-step""",
        default=None,
    )
    convection: Optional[Bloc_convection] = Field(
        description=r"""Keyword to alter the convection scheme.""", default=None
    )
    diffusion: Optional[Bloc_diffusion] = Field(
        description=r"""Keyword to specify the diffusion operator.""", default=None
    )
    conditions_limites: Optional[Annotated[List[Condlimlu], "Condlims"]] = Field(
        description=r"""Boundary conditions.""", default=None
    )
    conditions_initiales: Optional[Annotated[List[Condinit], "Condinits"]] = Field(
        description=r"""Initial conditions.""", default=None
    )
    sources: Optional[Annotated[List[Source_base], "Sources"]] = Field(
        description=r"""The sources.""", default=None
    )
    ecrire_fichier_xyz_valeur: Optional[Ecrire_fichier_xyz_valeur] = Field(
        description=r"""This keyword is used to write the values of a field only for some boundaries in a text file""",
        default=None,
    )
    parametre_equation: Optional[Parametre_equation_base] = Field(
        description=r"""Keyword used to specify additional parameters for the equation""",
        default=None,
    )
    equation_non_resolue: Optional[str] = Field(
        description=r"""The equation will not be solved while condition(t) is verified if equation_non_resolue keyword is used. Exemple: The Navier-Stokes equations are not solved between time t0 and t1.  Navier_Sokes_Standard  { equation_non_resolue (t>t0)*(t<t1) }""",
        default=None,
    )
    renommer_equation: Optional[str] = Field(
        description=r"""Rename the equation with a specific name.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Transport_k_epsilon(Eqn_base):
    r"""
    The (k-eps) transport equation. To resume from a previous mixing length calculation, an
    external MED-format file containing reconstructed K and Epsilon quantities can be read
    (see fichier_ecriture_k_eps) thanks to the Champ_fonc_MED keyword.

    Warning, When used with the Quasi-compressible model, k and eps should be viewed as rho k
    and rho epsilon when defining initial and boundary conditions or when visualizing values
    for k and eps. This bug will be fixed in a future version.
    """

    with_nu: Optional[Literal["yes", "no"]] = Field(
        description=r"""yes/no""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "with_nu": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Transport_k(Eqn_base):
    r"""
    The k transport equation in bicephale (standard or realisable) k-eps model.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Transport_epsilon(Eqn_base):
    r"""
    The eps transport equation in bicephale (standard or realisable) k-eps model.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Mod_turb_hyd_rans(Modele_turbulence_hyd_deriv):
    r"""
    Class for RANS turbulence model for Navier-Stokes equations.
    """

    k_min: Optional[float] = Field(
        description=r"""Lower limitation of k (default value 1.e-10).""", default=None
    )
    quiet: Optional[bool] = Field(
        description=r"""To disable printing of information about K and Epsilon/Omega.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Loi_standard_hydr_old(Turbulence_paroi_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_expert_hydr(Loi_standard_hydr):
    r"""
    This keyword is similar to the previous keyword Loi_standard_hydr but has several
    additional options into brackets.
    """

    u_star_impose: Optional[float] = Field(
        description=r"""The value of the friction velocity (u*) is not calculated but given by the user.""",
        default=None,
    )
    methode_calcul_face_keps_impose: Optional[
        Literal["toutes_les_faces_accrochees", "que_les_faces_des_elts_dirichlet"]
    ] = Field(
        description=r"""The available options select the algorithm to apply K and Eps boundaries condition (the algorithms differ according to the faces). toutes_les_faces_accrochees : Default option in 2D (the algorithm is the same than the algorithm used in Loi_standard_hydr)  que_les_faces_des_elts_dirichlet : Default option in 3D (another algorithm where less faces are concerned when applying K-Eps boundary condition).""",
        default=None,
    )
    kappa: Optional[float] = Field(
        description=r"""The value can be changed from the default one (0.415)""",
        default=None,
    )
    erugu: Optional[float] = Field(
        description=r"""The value of E can be changed from the default one for a smooth wall (9.11). It is also possible to change the value for one boundary wall only with paroi_rugueuse keyword/""",
        default=None,
    )
    a_plus: Optional[float] = Field(
        description=r"""The value can can be changed from the default one (26.0)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "u_star_impose": [],
        "methode_calcul_face_keps_impose": [],
        "kappa": [],
        "erugu": [],
        "a_plus": [],
    }


################################################################


class Loi_standard_hydr_scalaire(Turbulence_paroi_scalaire_base):
    r"""
    Keyword for the law of the wall.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_expert_scalaire(Loi_standard_hydr_scalaire):
    r"""
    Keyword similar to keyword Loi_standard_hydr_scalaire but with additional option.
    """

    prdt_sur_kappa: Optional[float] = Field(
        description=r"""This option is to change the default value of 2.12 in the scalable wall function.""",
        default=None,
    )
    calcul_ldp_en_flux_impose: Optional[Literal[0, 1]] = Field(
        description=r"""By default (value set to 0), the law of the wall is not applied for a wall with a Neumann condition. With value set to 1, the law is applied even on a wall with Neumann condition.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "prdt_sur_kappa": [],
        "calcul_ldp_en_flux_impose": [],
    }


################################################################


class Loi_analytique_scalaire(Turbulence_paroi_scalaire_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Twofloat(Objet_lecture):
    r"""
    two reals.
    """

    a: float = Field(description=r"""First real.""", default=0.0)
    b: float = Field(description=r"""Second real.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "a": [], "b": []}


################################################################


class Liste_sonde_tble(Listobj):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Un_point(Objet_lecture):
    r"""
    A point.
    """

    pos: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Point coordinates.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "pos": []}


################################################################


class Sonde_tble(Objet_lecture):
    r"""
    not_set
    """

    name: str = Field(description=r"""not_set""", default="")
    point: Un_point = Field(
        description=r"""not_set""", default_factory=lambda: eval("Un_point()")
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "name": [], "point": []}


################################################################


class Paroi_tble(Turbulence_paroi_base):
    r"""
    Keyword for the Thin Boundary Layer Equation wall-model (a more complete description of
    the model can be found into this PDF file). The wall shear stress is evaluated thanks to
    boundary layer equations applied in a one-dimensional fine grid in the near-wall region.
    """

    n: Optional[int] = Field(
        description=r"""Number of nodes in the TBLE grid (mandatory option).""",
        default=None,
    )
    facteur: Optional[float] = Field(
        description=r"""Stretching ratio for the TBLE grid (to refine, the TBLE facteur must be greater than 1).""",
        default=None,
    )
    modele_visco: Optional[str] = Field(
        description=r"""File name containing the description of the eddy viscosity model.""",
        default=None,
    )
    stats: Optional[Twofloat] = Field(
        description=r"""Statistics of the TBLE velocity and turbulent viscosity profiles. 2 values are required : the starting time and ending time of the statistics computation.""",
        default=None,
    )
    sonde_tble: Optional[Annotated[List[Sonde_tble], "Liste_sonde_tble"]] = Field(
        description=r"""not_set""", default=None
    )
    restart: Optional[bool] = Field(description=r"""not_set""", default=None)
    stationnaire: Optional[Floatfloat] = Field(description=r"""not_set""", default=None)
    lambda_: Optional[str] = Field(description=r"""not_set""", default=None)
    mu: Optional[str] = Field(description=r"""not_set""", default=None)
    sans_source_boussinesq: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    alpha: Optional[float] = Field(description=r"""not_set""", default=None)
    kappa: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "n": [],
        "facteur": [],
        "modele_visco": [],
        "stats": [],
        "sonde_tble": [],
        "restart": [],
        "stationnaire": [],
        "lambda_": ["lambda_u", "lambda"],
        "mu": [],
        "sans_source_boussinesq": [],
        "alpha": [],
        "kappa": [],
    }


################################################################


class Fourfloat(Objet_lecture):
    r"""
    Four reals.
    """

    a: float = Field(description=r"""First real.""", default=0.0)
    b: float = Field(description=r"""Second real.""", default=0.0)
    c: float = Field(description=r"""Third real.""", default=0.0)
    d: float = Field(description=r"""Fourth real.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "a": [], "b": [], "c": [], "d": []}


################################################################


class Paroi_tble_scal(Turbulence_paroi_scalaire_base):
    r"""
    Keyword for the Thin Boundary Layer Equation thermal wall-model.
    """

    n: Optional[int] = Field(
        description=r"""Number of nodes in the TBLE grid (mandatory option).""",
        default=None,
    )
    facteur: Optional[float] = Field(
        description=r"""Stretching ratio for the TBLE grid (to refine, the TBLE facteur must be greater than 1).""",
        default=None,
    )
    modele_visco: Optional[str] = Field(
        description=r"""File name containing the description of the eddy viscosity model.""",
        default=None,
    )
    nb_comp: Optional[int] = Field(
        description=r"""Number of component to solve in the fine grid (1 if 2D simulation (2D not available yet), 2 if 3D simulation).""",
        default=None,
    )
    stats: Optional[Fourfloat] = Field(
        description=r"""Statistics of the TBLE velocity and turbulent viscosity profiles. 4 values are required : the starting time of velocity averaging, the starting time of the RMS fluctuations, the ending time of the statistics computation and finally the print time period for the statistics.""",
        default=None,
    )
    sonde_tble: Optional[Annotated[List[Sonde_tble], "Liste_sonde_tble"]] = Field(
        description=r"""not_set""", default=None
    )
    prandtl: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "n": [],
        "facteur": [],
        "modele_visco": [],
        "nb_comp": [],
        "stats": [],
        "sonde_tble": [],
        "prandtl": [],
    }


################################################################


class Loi_paroi_nu_impose(Turbulence_paroi_scalaire_base):
    r"""
    Keyword to impose Nusselt numbers on the wall for the thermohydraulic problems. To use
    this option, it is necessary to give in the data file the value of the hydraulic diameter
    and the expression of the Nusselt number.
    """

    nusselt: str = Field(
        description=r"""The Nusselt number. This expression can be a function of x, y, z, Re (Reynolds number), Pr (Prandtl number).""",
        default="",
    )
    diam_hydr: Field_base = Field(
        description=r"""The hydraulic diameter.""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nusselt": [], "diam_hydr": []}


################################################################


class Utau_imp(Turbulence_paroi_base):
    r"""
    Keyword to impose the friction velocity on the wall with a turbulence model for
    thermohydraulic problems. There are two possibilities to use this keyword :

    1 - we can impose directly the value of the friction velocity u_star.

    2 - we can also give the friction coefficient and hydraulic diameter. So, TRUST
    determines the friction velocity by : u_star = U*sqrt(lambda_c/8).
    """

    u_tau: Optional[Field_base] = Field(description=r"""Field type.""", default=None)
    lambda_c: Optional[str] = Field(
        description=r"""The friction coefficient. It can be function of the spatial coordinates x,y,z, the Reynolds number Re, and the hydraulic diameter.""",
        default=None,
    )
    diam_hydr: Optional[Field_base] = Field(
        description=r"""The hydraulic diameter.""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "u_tau": [], "lambda_c": [], "diam_hydr": []}


################################################################


class Source_transport_eps(Source_base):
    r"""
    Keyword to alter the source term constants for eps in the bicephale k-eps model epsilon
    transport equation. By default, these constants are set to: C1_eps=1.44 C2_eps=1.92
    """

    c1_eps: Optional[float] = Field(description=r"""First constant.""", default=None)
    c2_eps: Optional[float] = Field(description=r"""Second constant.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "c1_eps": [], "c2_eps": []}


################################################################


class Source_transport_k(Source_base):
    r"""
    Keyword to alter the source term constants for k in the bicephale k-eps model epsilon
    transport equation.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Source_transport_k_eps(Source_base):
    r"""
    Keyword to alter the source term constants in the standard k-eps model epsilon transport
    equation. By default, these constants are set to: C1_eps=1.44 C2_eps=1.92
    """

    c1_eps: Optional[float] = Field(description=r"""First constant.""", default=None)
    c2_eps: Optional[float] = Field(description=r"""Second constant.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "c1_eps": [], "c2_eps": []}


################################################################


class Source_transport_k_eps_anisotherme(Source_transport_k_eps):
    r"""
    Keywords to modify the source term constants in the anisotherm standard k-eps model
    epsilon transport equation. By default, these constants are set to: C1_eps=1.44
    C2_eps=1.92 C3_eps=1.0
    """

    c3_eps: Optional[float] = Field(description=r"""Third constant.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "c3_eps": [], "c1_eps": [], "c2_eps": []}


################################################################


class Source_transport_k_eps_aniso_concen(Source_transport_k_eps):
    r"""
    Keywords to modify the source term constants in the anisotherm standard k-eps model
    epsilon transport equation. By default, these constants are set to: C1_eps=1.44
    C2_eps=1.92 C3_eps=1.0
    """

    c3_eps: Optional[float] = Field(description=r"""Third constant.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "c3_eps": [], "c1_eps": [], "c2_eps": []}


################################################################


class Source_transport_k_eps_aniso_therm_concen(Source_transport_k_eps):
    r"""
    Keywords to modify the source term constants in the anisotherm standard k-eps model
    epsilon transport equation. By default, these constants are set to: C1_eps=1.44
    C2_eps=1.92 C3_eps=1.0
    """

    c3_eps: Optional[float] = Field(description=r"""Third constant.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "c3_eps": [], "c1_eps": [], "c2_eps": []}


################################################################


class Frontiere_ouverte_k_eps_impose(Condlim_base):
    r"""
    Turbulence condition imposed on an open boundary called bord (edge) (this situation
    corresponds to a fluid inlet). This condition must be associated with an imposed inlet
    velocity condition.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_k_omega_impose(Condlim_base):
    r"""
    Turbulence condition imposed on an open boundary called bord (edge) (this situation
    corresponds to a fluid inlet). This condition must be associated with an imposed inlet
    velocity condition.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Boundary_field_uniform_keps_from_ud(Front_field_base):
    r"""
    field which allows to impose on a boundary K and EPS values derived from U velocity and D
    hydraulic diameter
    """

    u: float = Field(description=r"""value of velocity""", default=0.0)
    d: float = Field(description=r"""value of hydraulic diameter""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "u": [], "d": []}


################################################################


class Field_uniform_keps_from_ud(Field_base):
    r"""
    field which allows to impose on a domain K and EPS values derived from U velocity and D
    hydraulic diameter
    """

    u: float = Field(
        description=r"""value of velocity specified in boundary condition.""",
        default=0.0,
    )
    d: float = Field(
        description=r"""value of hydraulic diameter specified in boundary condition""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {None: [], "u": [], "d": []}


################################################################


class Dirichlet(Condlim_base):
    r"""
    Dirichlet condition at the boundary called bord (edge) : 1). For Navier-Stokes equations,
    velocity imposed at the boundary; 2). For scalar transport equation, scalar imposed at the
    boundary.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_rugueuse(Dirichlet):
    r"""
    Rough wall boundary
    """

    erugu: float = Field(description=r"""Constant value for roughness""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "erugu": []}


################################################################


class Diffusion_tenseur_reynolds_externe(Diffusion_deriv):
    r"""
    Estimate the values of the Reynolds tensor.
    """

    _synonyms: ClassVar[dict] = {None: ["tenseur_reynolds_externe"]}


################################################################


class Bloc_lecture_turb_synt(Objet_lecture):
    r"""
    bloc containing parameters of the synthetic turbulence
    """

    moyenne: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""components of the average velocity fields""",
        default_factory=list,
    )
    lenghtscale: float = Field(description=r"""turbulent length scale""", default=0.0)
    nbmodes: int = Field(description=r"""number of Fourier modes""", default=0)
    turbkinen: float = Field(
        description=r"""turbulent kinetic energy (k)""", default=0.0
    )
    turbdissrate: float = Field(
        description=r"""turbulent dissipation rate (epsilon)""", default=0.0
    )
    ratiocutoffwavenumber: float = Field(
        description=r"""ratio between the cut-off wavenumber and pi/delta""",
        default=0.0,
    )
    keoverkmin: float = Field(
        description=r"""ratio of the most energetic wavenumber Ke over the minimum wavenumber Kmin representing the largest turbulent eddies""",
        default=0.0,
    )
    timescale: float = Field(description=r"""turbulent time scale""", default=0.0)
    dir_fluct: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""directions for the velocity fluctations (e.g 1 0 0 generates velocity fluctuations in the x-direction only)""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "moyenne": [],
        "lenghtscale": ["lengthscale"],
        "nbmodes": [],
        "turbkinen": [],
        "turbdissrate": [],
        "ratiocutoffwavenumber": [],
        "keoverkmin": [],
        "timescale": [],
        "dir_fluct": [],
    }


################################################################


class Champ_front_synt(Front_field_base):
    r"""
    Boundary condition to create the synthetic fluctuations as inlet boundary. Available only
    for 3D configurations.
    """

    dim: int = Field(
        description=r"""Number of field components. It should be 3!""", default=0
    )
    bloc: Bloc_lecture_turb_synt = Field(
        description=r"""bloc containing the parameters of the synthetic turbulence""",
        default_factory=lambda: eval("Bloc_lecture_turb_synt()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "bloc": []}


################################################################


class Convection_diffusion_concentration(Eqn_base):
    r"""
    Constituent transport vectorial equation (concentration diffusion convection).
    """

    nom_inconnue: Optional[str] = Field(
        description=r"""Keyword Nom_inconnue will rename the unknown of this equation with the given name. In the postprocessing part, the concentration field will be accessible with this name. This is usefull if you want to track more than one concentration (otherwise, only the concentration field in the first concentration equation can be accessed).""",
        default=None,
    )
    alias: Optional[str] = Field(description=r"""not_set""", default=None)
    masse_molaire: Optional[float] = Field(description=r"""not_set""", default=None)
    is_multi_scalar: Optional[bool] = Field(
        description=r"""Flag to activate the multi_scalar diffusion operator""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nom_inconnue": [],
        "alias": [],
        "masse_molaire": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_concentration_turbulent(Convection_diffusion_concentration):
    r"""
    Constituent transport equations (concentration diffusion convection) as well as the
    associated turbulence model equations.
    """

    modele_turbulence: Optional[Modele_turbulence_scal_base] = Field(
        description=r"""Turbulence model to be used in the constituent transport equations. The only model currently available is Schmidt.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "nom_inconnue": [],
        "alias": [],
        "masse_molaire": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_concentration_turbulent_ft_disc(
    Convection_diffusion_concentration_turbulent
):
    r"""
    equation_non_resolue
    """

    equation_interface: Optional[str] = Field(
        description=r"""his is the name of the interface tracking equation to watch. The scalar will not diffuse through the interface of this equation.""",
        default=None,
    )
    phase: Literal[0, 1] = Field(
        description=r"""tells whether the scalar must be confined in phase 0 or in phase 1""",
        default=0,
    )
    option: Optional[str] = Field(
        description=r"""Experimental features used to prevent the concentration to leak through the interface between phases due to numerical diffusion.  RIEN: do nothing  RAMASSE_MIETTES_SIMPLE: at each timestep, this algorithm takes all the mass located in the opposite phase and spreads it uniformly in the given phase.""",
        default=None,
    )
    equations_source_chimie: Optional[List[str]] = Field(
        description=r"""This term specifies the name of the concentration equation of the reagents. It should be specified only in the bloc that concerns the convection/diffusion equation of the product.""",
        default=None,
    )
    modele_cinetique: Optional[int] = Field(
        description=r"""This is the keyword that the user defines for the reaction model that he wants to use. Four reaction models are currently offered (1 to 4). Model 1 is the default one and is based on the laminar rate formulation. Model 2 employs an LES diffusive EDC formulation. Model 3 defines an LES variance formulation. Model 4 is a mix between models 2 and 3.""",
        default=None,
    )
    equation_nu_t: Optional[str] = Field(
        description=r"""This specifies the name of the hydraulic equation used which defines the turbulent (basically SGS) viscosity.""",
        default=None,
    )
    constante_cinetique: Optional[float] = Field(
        description=r"""This is the constant kinetic rate of the reaction and is used for the laminar model 1 only.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "equation_interface": [],
        "phase": [],
        "option": [],
        "equations_source_chimie": [],
        "modele_cinetique": [],
        "equation_nu_t": [],
        "constante_cinetique": [],
        "modele_turbulence": [],
        "nom_inconnue": [],
        "alias": [],
        "masse_molaire": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Source_constituant_vortex(Source_base):
    r"""
    Special treatment for the reactor of vortex effect where reagents are injected just below
    the free surface in the liquid phase
    """

    senseur_interface: Optional[Bloc_lecture] = Field(
        description=r"""This is to be defined for the concentration equation of the reagents only and in the bloc of the sources. Here the user defines the position of the reagents injection.""",
        default=None,
    )
    rayon_spot: Optional[float] = Field(
        description=r"""defines the radius of the concentration spot (tracer) injected in the fluid""",
        default=None,
    )
    delta_spot: Optional[List[float]] = Field(
        description=r"""dimensions of the injection (segment). the syntax is dim val1 val2 [val3]""",
        default=None,
    )
    integrale: Optional[float] = Field(
        description=r"""the molar flowrate of injection""", default=None
    )
    debit: Optional[float] = Field(
        description=r"""a normalization of the molar flow rate. Advice: keep this value to 1.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "senseur_interface": [],
        "rayon_spot": [],
        "delta_spot": [],
        "integrale": [],
        "debit": [],
    }


################################################################


class Postraitement_base(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Postraitement_ft_lata(Postraitement_base):
    r"""
    not_set
    """

    bloc: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Residuals(Interprete):
    r"""
    To specify how the residuals will be computed.
    """

    norm: Optional[Literal["l2", "max"]] = Field(
        description=r"""allows to choose the norm we want to use (max norm by default). Possible to specify L2-norm.""",
        default=None,
    )
    relative: Optional[Literal["0", "1", "2"]] = Field(
        description=r"""This is the old keyword seuil_statio_relatif_deconseille. If it is set to 1, it will normalize the residuals with the residuals of the first 5 timesteps (default is 0). if set to 2, residual will be computed as R/(max-min).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "norm": [], "relative": []}


################################################################


class Dt_start(Class_generic):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Schema_temps_base(Objet_u):
    r"""
    Basic class for time schemes. This scheme will be associated with a problem and the
    equations of this problem.
    """

    tinit: Optional[float] = Field(
        description=r"""Value of initial calculation time (0 by default).""",
        default=None,
    )
    tmax: Optional[float] = Field(
        description=r"""Time during which the calculation will be stopped (1e30s by default).""",
        default=None,
    )
    tcpumax: Optional[float] = Field(
        description=r"""CPU time limit (must be specified in hours) for which the calculation is stopped (1e30s by default).""",
        default=None,
    )
    dt_min: Optional[float] = Field(
        description=r"""Minimum calculation time step (1e-16s by default).""",
        default=None,
    )
    dt_max: Optional[str] = Field(
        description=r"""Maximum calculation time step as function of time (1e30s by default).""",
        default=None,
    )
    dt_sauv: Optional[float] = Field(
        description=r"""Save time step value (1e30s by default). Every dt_sauv, fields are saved in the .sauv file. The file contains all the information saved over time. If this instruction is not entered, results are saved only upon calculation completion. To disable the writing of the .sauv files, you must specify 0. Note that dt_sauv is in terms of physical time (not cpu time).""",
        default=None,
    )
    dt_impr: Optional[float] = Field(
        description=r"""Scheme parameter printing time step in time (1e30s by default). The time steps and the flux balances are printed (incorporated onto every side of processed domains) into the .out file.""",
        default=None,
    )
    facsec: Optional[str] = Field(
        description=r"""Value assigned to the safety factor for the time step (1. by default). It can also be a function of time. The time step calculated is multiplied by the safety factor. The first thing to try when a calculation does not converge with an explicit time scheme is to reduce the facsec to 0.5.  Warning: Some schemes needs a facsec lower than 1 (0.5 is a good start), for example Schema_Adams_Bashforth_order_3.""",
        default=None,
    )
    seuil_statio: Optional[float] = Field(
        description=r"""Value of the convergence threshold (1e-12 by default). Problems using this type of time scheme converge when the derivatives dGi/dt  of all the unknown transported values Gi have a combined absolute value less than this value. This is the keyword used to set the permanent rating threshold.""",
        default=None,
    )
    residuals: Optional[Residuals] = Field(
        description=r"""To specify how the residuals will be computed (default max norm, possible to choose L2-norm instead).""",
        default=None,
    )
    diffusion_implicite: Optional[int] = Field(
        description=r"""Keyword to make the diffusive term in the Navier-Stokes equations implicit (in this case, it should be set to 1). The stability time step is then only based on the convection time step (dt=facsec*dt_convection). Thus, in some circumstances, an important gain is achieved with respect to the time step (large diffusion with respect to convection on tightened meshes). Caution: It is however recommended that the user avoids exceeding the convection time step by selecting a too large facsec value. Start with a facsec value of 1 and then increase it gradually if you wish to accelerate calculation. In addition, for a natural convection calculation with a zero initial velocity, in the first time step, the convection time is infinite and therefore dt=facsec*dt_max.""",
        default=None,
    )
    seuil_diffusion_implicite: Optional[float] = Field(
        description=r"""This keyword changes the default value (1e-6) of convergency criteria for the resolution by conjugate gradient used for implicit diffusion.""",
        default=None,
    )
    impr_diffusion_implicite: Optional[int] = Field(
        description=r"""Unactivate (default) or not the printing of the convergence during the resolution of the conjugate gradient.""",
        default=None,
    )
    impr_extremums: Optional[int] = Field(
        description=r"""Print unknowns extremas""", default=None
    )
    no_error_if_not_converged_diffusion_implicite: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    no_conv_subiteration_diffusion_implicite: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    dt_start: Optional[Dt_start] = Field(
        description=r"""dt_start dt_min : the first iteration is based on dt_min.  dt_start dt_calc : the time step at first iteration is calculated in agreement with CFL condition.  dt_start dt_fixe value : the first time step is fixed by the user (recommended when resuming calculation with Crank Nicholson temporal scheme to ensure continuity).  By default, the first iteration is based on dt_calc.""",
        default=None,
    )
    nb_pas_dt_max: Optional[int] = Field(
        description=r"""Maximum number of calculation time steps (1e9 by default).""",
        default=None,
    )
    niter_max_diffusion_implicite: Optional[int] = Field(
        description=r"""This keyword changes the default value (number of unknowns) of the maximal iterations number in the conjugate gradient method used for implicit diffusion.""",
        default=None,
    )
    precision_impr: Optional[int] = Field(
        description=r"""Optional keyword to define the digit number for flux values printed into .out files (by default 3).""",
        default=None,
    )
    periode_sauvegarde_securite_en_heures: Optional[float] = Field(
        description=r"""To change the default period (23 hours) between the save of the fields in .sauv file.""",
        default=None,
    )
    no_check_disk_space: Optional[bool] = Field(
        description=r"""To disable the check of the available amount of disk space during the calculation.""",
        default=None,
    )
    disable_progress: Optional[bool] = Field(
        description=r"""To disable the writing of the .progress file.""", default=None
    )
    disable_dt_ev: Optional[bool] = Field(
        description=r"""To disable the writing of the .dt_ev file.""", default=None
    )
    gnuplot_header: Optional[int] = Field(
        description=r"""Optional keyword to modify the header of the .out files. Allows to use the column title instead of columns number.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_3(Schema_temps_base):
    r"""
    This is a low-storage Runge-Kutta scheme of third order that uses 3 integration points.
    The method is presented by Williamson (case 7) in
    https://www.sciencedirect.com/science/article/pii/0021999180900339
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Rk3_ft(Runge_kutta_ordre_3):
    r"""
    Keyword for Runge Kutta time scheme for Front_Tracking calculation.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Paroi_ft_disc_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_ft_disc_symetrie(Paroi_ft_disc_deriv):
    r"""
    Symetrie condition in the case of two-phase flows
    """

    _synonyms: ClassVar[dict] = {None: ["symetrie"]}


################################################################


class Paroi_ft_disc(Condlim_base):
    r"""
    Boundary condition for Front-Tracking problem in the discontinuous version.
    """

    type: Paroi_ft_disc_deriv = Field(
        description=r"""Symetrie condition.""",
        default_factory=lambda: eval("Paroi_ft_disc_deriv()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "type": []}


################################################################


class Paroi_ft_disc_constant(Paroi_ft_disc_deriv):
    r"""
    condition contact angle fidex. The angle is measured between the wall and the interface in
    the phase 0.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["constant"], "ch": []}


################################################################


class Paroi_echange_contact_vdf_ft(Paroi_echange_contact_vdf):
    r"""
    This boundary condition is used between a conduction problem and a thermohydraulic problem
    with two phases flow (Front-Tracking method) to modelize heat exchange.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "temp": [],
        "h": [],
    }


################################################################


class Milieu_v2_base(Objet_u):
    r"""
    Basic class for medium (physics properties of medium) composed of constituents (fluids and
    solids).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Methode_transport_deriv(Objet_lecture):
    r"""
    Basic class for method of transport of interface.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Vitesse_imposee(Methode_transport_deriv):
    r"""
    Class to specify that the speed of displacement of the nodes of the interfaces is imposed
    with an analytical formula.
    """

    val: Annotated[List[str], "size_is_dim"] = Field(
        description=r"""Analytical formula.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Vitesse_interpolee(Methode_transport_deriv):
    r"""
    Class to specify that the interpolation will use the velocity field of the Navier-Stokes
    equation named val to compute the speed of displacement of the nodes of the interfaces.
    """

    val: str = Field(description=r"""Navier-Stokes equation.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Bloc_lecture_remaillage(Objet_lecture):
    r"""
    Parameters for remeshing.
    """

    pas: Optional[float] = Field(
        description=r"""This keyword has default value -1.; when it is set to a negative value there is no remeshing. It is the time step in second (physical time) between two operations of remeshing.""",
        default=None,
    )
    pas_lissage: Optional[float] = Field(
        description=r"""This keyword has default value -1.; when it is set to a negative value there is no smoothing of mesh. It is the time step in second (physical time) between two operations of smoothing of the mesh.""",
        default=None,
    )
    nb_iter_remaillage: Optional[int] = Field(
        description=r"""This keyword has default value 0; when it is set to the zero value there is no remeshing. It is the number of iterations performed during a remeshing process.""",
        default=None,
    )
    nb_iter_barycentrage: Optional[int] = Field(
        description=r"""This keyword has default value 0; when it is set to the zero value there is no operation of barycentrage. The barycentrage operation consists in moving each node of the mesh tangentially to the mesh surface and in a direction that let it closer the center of gravity of its neighbors. If relax_barycentrage is set to 1, the node is move to the center of gravity. For values lower than unity, the motion is limited to the corresponding fraction. The parameter nb_iter_barycentrage is the number of iteration of these node displacements.""",
        default=None,
    )
    relax_barycentrage: Optional[float] = Field(
        description=r"""This keyword has default value 0; when it is set to the zero value there is no motion of the nodes. When 0 < relax_barycentrage <= 1, this parameter provides the relaxation ratio to be used in the barycentrage operation described for the keyword nb_iter_barycentrage.""",
        default=None,
    )
    critere_arete: Optional[float] = Field(
        description=r"""This keyword is used to compute two sub-criteria : the minimum and the maximum edge length ratios used in the process of obtaining edges of length close to critere_longueur_fixe. Their respective values are set to (1-critere_arete)**2 and (1+critere_arete)**2. The default values of the minimum and the maximum are set respectively to 0.5 and 1.5. When an edge is longer than critere_longueur_fixe*(1+critere_arete)**2, the edge is cut into two pieces; when its length is smaller than critere_longueur_fixe*(1-critere_arete)**2, this edge has to be suppressed.""",
        default=None,
    )
    critere_remaillage: Optional[float] = Field(
        description=r"""This keyword was previously used to compute two sub-criteria : the minimum and the maximum length used in the process of remeshing. Their respective values are set to (1-critere_remaillage)**2 and (1+critere_remaillage)**2. The default values of the minimum and the maximum are set respectively to 0.2 and 1.7. There are currently not used in data files.""",
        default=None,
    )
    impr: Optional[float] = Field(
        description=r"""This keyword is followed by a value that specify the printing time period given. The default value is -1, which means no printing.""",
        default=None,
    )
    facteur_longueur_ideale: Optional[float] = Field(
        description=r"""This keyword is used to set a ratio between edge length and the cube root of volume cell for the remeshing process. The default value is 1.0.""",
        default=None,
    )
    nb_iter_correction_volume: Optional[int] = Field(
        description=r"""This keyword give the maximum number of iterations to be performed trying to satisfy the criterion seuil_dvolume_residuel. The default value is 0, which means no iteration.""",
        default=None,
    )
    seuil_dvolume_residuel: Optional[float] = Field(
        description=r"""This keyword give the error volume (in m3) that is accepted to stop the iterations performed to keep the volume constant during the remeshing process. The default value is 0.0.""",
        default=None,
    )
    lissage_courbure_coeff: Optional[float] = Field(
        description=r"""This keyword is used to specify the diffusion coefficient used in the diffusion process of the curvature in the curvature smoothing process with a time step. The default value is 0.05. That value usually provides a stable process. Too small values do not stabilize enough the interface, especially with several Lagrangian nodes per Eulerian cell. Too high values induce an additional macroscopic smoothing of the interface that should physically come from the surface tension and not from this numerical smoothing.""",
        default=None,
    )
    lissage_courbure_iterations: Optional[int] = Field(
        description=r"""This keyword is used to specify the number of iterations to perform the curvature smoothing process. The default value is 1.""",
        default=None,
    )
    lissage_courbure_iterations_systematique: Optional[int] = Field(
        description=r"""These keywords allow a finer control than the previous lissage_courbure_iterations keyword. N1 iterations are applied systematically at each timestep. For proper DNS computation, N1 should be set to 0.""",
        default=None,
    )
    lissage_courbure_iterations_si_remaillage: Optional[int] = Field(
        description=r"""N2 iterations are applied only if the local or the global remeshing effectively changes the lagrangian mesh connectivity.""",
        default=None,
    )
    critere_longueur_fixe: Optional[float] = Field(
        description=r"""This keyword is used to specify the ideal edge length for a remeshing process. The default value is -1., which means that the remeshing does not try to have all edge lengths to tend towards a given value.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "pas": [],
        "pas_lissage": [],
        "nb_iter_remaillage": [],
        "nb_iter_barycentrage": [],
        "relax_barycentrage": [],
        "critere_arete": [],
        "critere_remaillage": [],
        "impr": [],
        "facteur_longueur_ideale": [],
        "nb_iter_correction_volume": [],
        "seuil_dvolume_residuel": [],
        "lissage_courbure_coeff": [],
        "lissage_courbure_iterations": [],
        "lissage_courbure_iterations_systematique": [],
        "lissage_courbure_iterations_si_remaillage": [],
        "critere_longueur_fixe": [],
    }


################################################################


class Convection_diffusion_concentration_ft_disc(Convection_diffusion_concentration):
    r"""
    not_set
    """

    equation_interface: Optional[str] = Field(
        description=r"""his is the name of the interface tracking equation to watch. The scalar will not diffuse through the interface of this equation.""",
        default=None,
    )
    phase: Literal[0, 1] = Field(
        description=r"""tells whether the scalar must be confined in phase 0 or in phase 1""",
        default=0,
    )
    option: Optional[str] = Field(
        description=r"""Experimental features used to prevent the concentration to leak through the interface between phases due to numerical diffusion.  RIEN: do nothing  RAMASSE_MIETTES_SIMPLE: at each timestep, this algorithm takes all the mass located in the opposite phase and spreads it uniformly in the given phase.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "equation_interface": [],
        "phase": [],
        "option": [],
        "nom_inconnue": [],
        "alias": [],
        "masse_molaire": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Objet_lecture_maintien_temperature(Objet_lecture):
    r"""
    not_set
    """

    sous_zone: str = Field(description=r"""not_set""", default="")
    temperature_moyenne: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "sous_zone": [],
        "temperature_moyenne": [],
    }


################################################################


class Penalisation_l2_ftd(Listobj):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["pp"]}


################################################################


class Penalisation_l2_ftd_lec(Objet_lecture):
    r"""
    not_set
    """

    postraiter_gradient_pression_sans_masse: Optional[int] = Field(
        description=r"""(IBM advanced) avoid mass matrix multiplication for the gradient postprocessing""",
        default=None,
    )
    correction_matrice_projection_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix matrix of initial projection for PDF""",
        default=None,
    )
    correction_calcul_pression_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix initial pressure computation for PDF""",
        default=None,
    )
    correction_vitesse_projection_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix initial velocity computation for PDF""",
        default=None,
    )
    correction_matrice_pression: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure matrix for PDF""", default=None
    )
    matrice_pression_penalisee_h1: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure matrix for PDF""", default=None
    )
    correction_vitesse_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix velocity for PDF""", default=None
    )
    correction_pression_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure for PDF""", default=None
    )
    gradient_pression_qdm_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure gradient""", default=None
    )
    bord: str = Field(description=r"""not_set""", default="")
    val: List[float] = Field(description=r"""not_set""", default_factory=list)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "postraiter_gradient_pression_sans_masse": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "matrice_pression_penalisee_h1": [],
        "correction_vitesse_modifie": [],
        "correction_pression_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "bord": [],
        "val": [],
    }


################################################################


class Convection_diffusion_temperature(Eqn_base):
    r"""
    Energy equation (temperature diffusion convection).
    """

    penalisation_l2_ftd: Optional[
        Annotated[List[Penalisation_l2_ftd_lec], "Penalisation_l2_ftd"]
    ] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "penalisation_l2_ftd": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_temperature_ft_disc(Convection_diffusion_temperature):
    r"""
    not_set
    """

    equation_interface: Optional[str] = Field(
        description=r"""The name of the interface equation should be given.""",
        default=None,
    )
    phase: Literal[0, 1] = Field(
        description=r"""Phase in which the temperature equation will be solved. The temperature, which may be postprocessed with the keyword temperature_EquationName, in the orther phase may be negative: the code only computes the temperature field in the specified phase. The other phase is supposed to physically stay at saturation temperature. The code uses a ghost fluid numerical method to work on a smooth temperature field at the interface. In the opposite phase (1-X) the temperature will therefore be extrapolated in the vicinity of the interface and have the opposite sign, saturation temperature is zero by convention).""",
        default=0,
    )
    equation_navier_stokes: Optional[str] = Field(
        description=r"""The name of the Navier Stokes equation of the problem should be given.""",
        default=None,
    )
    stencil_width: Optional[int] = Field(
        description=r"""distance in mesh elements over which the temperature field should be extrapolated in the opposite phase.""",
        default=None,
    )
    maintien_temperature: Optional[Objet_lecture_maintien_temperature] = Field(
        description=r"""maintien_temperature SOUS_ZONE_NAME VALUE : experimental, this acts as a dynamic source term that heats or cools the fluid to maintain the average temperature to VALUE within the specified region. At this time, this is done by multiplying the temperature within the SOUS_ZONE by an appropriate uniform value at each timestep. This feature might be implemented in a separate source term in the future.""",
        default=None,
    )
    prescribed_mpoint: Optional[float] = Field(
        description=r"""User defined value of the phase-change rate (override the value computed based on the temperature field)""",
        default=None,
    )
    correction_mpoint_diff_conv_energy: Optional[List[float]] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "equation_interface": [],
        "phase": [],
        "equation_navier_stokes": [],
        "stencil_width": [],
        "maintien_temperature": [],
        "prescribed_mpoint": [],
        "correction_mpoint_diff_conv_energy": [],
        "penalisation_l2_ftd": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Interpolation_champ_face_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_indic_faces_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_indic_faces_standard(Type_indic_faces_deriv):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["standard"]}


################################################################


class Type_indic_faces_modifiee(Type_indic_faces_deriv):
    r"""
    not_set
    """

    position: Optional[float] = Field(description=r"""not_set""", default=None)
    thickness: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["modifiee"], "position": [], "thickness": []}


################################################################


class Type_indic_faces_ai_based(Type_indic_faces_deriv):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["ai_based"]}


################################################################


class Interpolation_champ_facebase(Interpolation_champ_face_deriv):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["base"]}


################################################################


class Interpolation_champ_face_lineaire(Interpolation_champ_face_deriv):
    r"""
    not_set
    """

    vitesse_fluide_explicite: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["lineaire"], "vitesse_fluide_explicite": []}


################################################################


class Parcours_interface(Objet_lecture):
    r"""
    allows you to configure the algorithm that computes the surface mesh to volume mesh
    intersection. This algorithm has some serious trouble when the surface mesh points
    coincide with some faces of the volume mesh. Effects are visible on the indicator
    function, in VDF when a plane interface coincides with a volume mesh surface.

    To overcome these problems, the keyword correction_parcours_thomas keyword can be used:
    it allows the algorithm to slightly move some mesh points. This algorithm, which is
    experimental and is NOT activated by default, triggers a correction that avoids some
    errors in the computation of the indicator function for surface meshes that exactly cross
    some eulerian mesh edges (strongly suggested !).
    """

    correction_parcours_thomas: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "correction_parcours_thomas": []}


################################################################


class Transport_interfaces_ft_disc(Eqn_base):
    r"""
    Interface tracking equation for Front-Tracking problem in the discontinuous version.
    """

    conditions_initiales: Optional[Bloc_lecture] = Field(
        description=r"""The keyword conditions_initiales is used to define the shape of the initial interfaces through the zero level-set of a function, or through a mesh fichier_geom. Indicator function is set to 0, that is fluide0, where the function is negative; indicator function is set to 1, that is fluide1, where the function is positive; the interfaces are the level-set 0 of that function:   conditions_initiales { fonction  $(-((x-0.002)^2+(y-0.002)^2+z^2-(0.00125)^2))*((x-0.005)^2+(y-0.007)^2+z^2 (0.00150)^2))*(0.020-z))$  }   In the above example, there are three interfaces: two bubbles in a liquid with a free surface. One bubble has a radius of 0.00125, i.e. 1.25 mm, and its center is {0.002, 0.002, 0.000}. The other bubble has a radius of 0.00150, i.e. 1.5 mm, and its center is {0.005, 0.007, 0.000}. The free surface is above the two bubble, at a level z=0.02.   Additional feature in this block concerns the keywords ajout_phase0 and ajout_phase1. They can be used to simplify the composition of different interfaces. When using these keywords, the initial function defines the indicator function; ajout_phase0 and ajout_phase1 are used to modify this initial field. Each time ajout_phase0 is used, the field is untouched where the function is positive whereas the indicator field is set to 0 where the function is negative. The keyword ajout_phase1 has the symmetrical use, keeping the field value where the function is negative and setting the indicator field to 1 where the function is positive. The previous example can also be written:   conditions_initiales {  fonction z-0.020 , NL fonction ajout_phase1 $(x-0.002)^2+(y-0.002)^2+z^2-(0.00125)^2$ ,  fonction ajout_phase1 $(x-0.005)^2+(y-0.007)^2+z^2-(0.00150)^2$  }""",
        default=None,
    )
    methode_transport: Optional[Methode_transport_deriv] = Field(
        description=r"""Method of transport of interface.""", default=None
    )
    iterations_correction_volume: Optional[int] = Field(
        description=r"""Keyword to specify the number or iterations requested for the correction process that can be used to keep the volume of the phases constant during the transport process.""",
        default=None,
    )
    n_iterations_distance: Optional[int] = Field(
        description=r"""Keyword to specify the number or iterations requested for the smoothing process of computing the field corresponding to the signed distance to the interfaces and located at the center of the Eulerian elements. This smoothing is necessary when there are more Lagrangian nodes than Eulerian two-phase cells.""",
        default=None,
    )
    maillage: Optional[str] = Field(
        description=r"""This optional block is used to specify that we want a Gnuplot drawing of the initial mesh. There is only one keyword, niveau_plot, that is used only to define if a Gnuplot drawing is active (value 1) or not active (value -1). By default, skipping the block will produce non Gnuplot drawing. This option is to be used only in a debug process.""",
        default=None,
    )
    remaillage: Optional[Bloc_lecture_remaillage] = Field(
        description=r"""This block is used to specify the operations that are used to keep the solid interfaces in a proper condition. The remaillage block only contains parameter\'s values.""",
        default=None,
    )
    collisions: Optional[str] = Field(
        description=r"""This block is used to specify the operations that are used when a collision occurs between two parts of interfaces. When this occurs, it is necessary to build a new mesh that has locally a clear definition of what is inside and what is outside of the mesh. The collisions can either be active or inactive. If the collisions are active (highly recommended), a Juric level-set reconstruction method will be used to re-create the new mesh after each coalescence or breakup. An option Juric_local phase_continue N can be used to force the remeshing to impact only a local portion of the mesh, near the collision. The next line (type_remaillage) is used to state whose field will be used for the level-set computation. Main option is Juric, a remeshing that is compatible with parallel computing. When using Juric level-set remeshing, the source field (source_isovaleur) that is used to compute the level-sets is then defined. It can be either the indicator function (indicatrice), a choice which is the default one and the most robust, or a geometrical distance computed from the mesh at the beginning of the time step (fonction_distance), a choice that may be more accurate in specific situations. Type_remaillage can be either Juric or Thomas. When Thomas is used, it is an enhancement of the Juric remeshing algorithm designed to compensate for mass loss during remeshing. The mesh is always reconstructed with the indicator function (not with the distance function). After having reconstructed the mesh with the Juric algorithm, the difference between the old indicator function (before remeshing) and the new indicator function is computed. The differences occuring at a distance below or equal to N elements from the interface are summed up and used to move the interface in the normal direction. The displacement of the interface is such that the volume of each phase after displacement is equal to the volume of the phase before remeshing. N (default value 1) must be smaller than n_iterations_distance (suggested value: 2).""",
        default=None,
    )
    methode_interpolation_v: Optional[Literal["valeur_a_elem", "vdf_lineaire"]] = Field(
        description=r"""In this block, two keywords are possible for method to select the way the interpolation is performed. With the choice valeur_a_elem the speed of displacement of the nodes of the interfaces is the velocity at the center of the Eulerian element in which each node is located at the beginning of the time step. This choice is the default interpolation method. The choice VDF_lineaire is only available with a VDF discretization (VDF). In this case, the speed of displacement of the nodes of the interfaces is linearly interpolated on the 4 (in 2D) or the 6 (in 3D) Eulerian velocities closest the location of each node at the beginning of the time step. In peculiar situation, this choice may provide a better interpolated value. Of course, this choice is not available with a VEF discretization (VEFPreP1B).""",
        default=None,
    )
    volume_impose_phase_1: Optional[float] = Field(
        description=r"""this keyword is used to specify the volume of one phase to keep the volume of the phases constant during the remeshing process. It is an alternate solution to trouble in mass conservation. This option is mainly realistic when only one inclusion of phase 1 is present in the domain. In most other situations, the iterations_correction_volume keyword seems easier to justify. The volume to be keep is in m3 and should agree with initial condition.""",
        default=None,
    )
    parcours_interface: Optional[Parcours_interface] = Field(
        description=r"""Parcours_interface allows you to configure the algorithm that computes the surface mesh to volume mesh intersection. This algorithm has some serious trouble when the surface mesh points coincide with some faces of the volume mesh. Effects are visible on the indicator function, in VDF when a plane interface coincides with a volume mesh surface.  To overcome these problems, the keyword correction_parcours_thomas keyword can be used: it allows the algorithm to slightly move some mesh points. This algorithm is experimental and is NOT activated by default.""",
        default=None,
    )
    interpolation_repere_local: Optional[bool] = Field(
        description=r"""Triggers a new transport algorithm for the interface: the velocity vector of lagrangian nodes is computed in the moving frame of reference of the center of each connex component, in such a way that relative displacements of nodes within a connex component of the lagrangian mesh are minimized, hence reducing the necessity of barycentering, smooting and local remeshing. Very efficient for bubbly flows.""",
        default=None,
    )
    interpolation_champ_face: Optional[Interpolation_champ_face_deriv] = Field(
        description=r"""It is possible to compute the imposed velocity for the solid-fluid interface by direct affectation (interpolation_scheme would be set to base) or by multi-linear interpolation (interpolation_scheme would be set to lineaire). The default value is base.""",
        default=None,
    )
    n_iterations_interpolation_ibc: Optional[int] = Field(
        description=r"""Useful only with interpolation_champ_face positioned to lineaire. Set the value concerning the width of the region of the linear interpolation. For the Penalized Direct Forcing model, a value equals to 1 is enough.""",
        default=None,
    )
    type_vitesse_imposee: Optional[Literal["uniforme", "analytique"]] = Field(
        description=r"""Useful only with interpolation_champ_face positioned to lineaire. Value of the keyword is uniforme (for an uniform solid-fluide interface\'s velocity, i.e. zero for instance) or analytique (for an analytic expression of the solid-fluide interface\'s velocity depending on the spatial coordinates). The default value is uniforme.""",
        default=None,
    )
    nombre_facettes_retenues_par_cellule: Optional[int] = Field(
        description=r"""Keyword to specify the default number (3) of facets per cell used to describe the geometry of the solid-solid interface. This number should be increased if the geometry of the solid-solid interface is complex in each cell (eulerian mesh too coarse for example).""",
        default=None,
    )
    seuil_convergence_uzawa: Optional[float] = Field(
        description=r"""Optional option to change the default value (10-8) of the threshold convergence for the Uzawa algorithm if used in the Penalized Direct Forcing model. Sometime, the value should be decreased to insure a better convergence to force equality between sequential and parallel results.""",
        default=None,
    )
    nb_iteration_max_uzawa: Optional[int] = Field(
        description=r"""Optional option to change the default value (10-8) of the threshold convergence for the Uzawa algorithm if used in the Penalized Direct Forcing model. Sometime, the value should be decreased to insure a better convergence to force equality between sequential and parallel results.""",
        default=None,
    )
    injecteur_interfaces: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    vitesse_imposee_regularisee: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    indic_faces_modifiee: Optional[Bloc_lecture] = Field(
        description=r"""not_set""", default=None
    )
    distance_projete_faces: Optional[Literal["simplifiee", "initiale", "modifiee"]] = (
        Field(description=r"""not_set""", default=None)
    )
    voflike_correction_volume: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    nb_lissage_correction_volume: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    nb_iterations_correction_volume: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    type_indic_faces: Optional[Type_indic_faces_deriv] = Field(
        description=r"""kind of interpolation to compute the face value of the phase indicator function (advanced option). Could be STANDARD, MODIFIEE or AI_BASED""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "conditions_initiales": ["initial_conditions"],
        "methode_transport": [],
        "iterations_correction_volume": [],
        "n_iterations_distance": [],
        "maillage": [],
        "remaillage": [],
        "collisions": [],
        "methode_interpolation_v": [],
        "volume_impose_phase_1": [],
        "parcours_interface": [],
        "interpolation_repere_local": [],
        "interpolation_champ_face": [],
        "n_iterations_interpolation_ibc": [],
        "type_vitesse_imposee": [],
        "nombre_facettes_retenues_par_cellule": [],
        "seuil_convergence_uzawa": [],
        "nb_iteration_max_uzawa": [],
        "injecteur_interfaces": [],
        "vitesse_imposee_regularisee": [],
        "indic_faces_modifiee": [],
        "distance_projete_faces": [],
        "voflike_correction_volume": [],
        "nb_lissage_correction_volume": [],
        "nb_iterations_correction_volume": [],
        "type_indic_faces": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Sortie_libre_rho_variable(Condlim_base):
    r"""
    Class to define an outlet boundary condition at which the pressure is defined through the
    given field, whereas the density of the two-phase flow may varies (value of P/rho given in
    Pa/kg.m-3).
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Injection_marqueur(Objet_lecture):
    r"""
    not_set
    """

    ensemble_points: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    proprietes_particules: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    t_debut_injection: Optional[float] = Field(description=r"""not_set""", default=None)
    dt_injection: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "ensemble_points": [],
        "proprietes_particules": [],
        "t_debut_injection": [],
        "dt_injection": [],
    }


################################################################


class Transport_marqueur_ft(Eqn_base):
    r"""
    not_set
    """

    conditions_initiales: Optional[Bloc_lecture] = Field(
        description=r"""ne semble pas standard""", default=None
    )
    injection: Optional[Injection_marqueur] = Field(
        description=r"""The keyword injection can be used to inject periodically during the calculation some other particles. The syntax for ensemble_points and proprietes_particles is the same than the initial conditions for the particles. The keyword t_debut_injection give the injection initial time (by default, given by t_debut_integration) and dt_injection gives the injection time period (by default given by dt_min).""",
        default=None,
    )
    transformation_bulles: Optional[Bloc_lecture] = Field(
        description=r"""This keyword will activate the transformation of an inclusion (small bubbles) into a particle. localisation gives the sub-zones (N number of sub-zones and their names) where the transformation may happen. The diameter size for the inclusion transformation is given by either diameter_min option, in this case the inclusion will be suppressed for a diameter less than diameter_size, either by the beta_transfo option, in this case the inclusion will be suppressed for a diameter less than diameter_size*cell_volume (cell_volume is the volume of the cell containing the inclusion). interface specifies the name of the inclusion interface and t_debut_transfo is the beginning time for the inclusion transformation operation (by default, it is t_debut_integr value) and dt_transfo is the period transformation (by default, it is dt_min value). In a two phase flow calculation, the particles will be suppressed when entring into the non marked phase""",
        default=None,
    )
    phase_marquee: Optional[int] = Field(
        description=r"""Phase number giving the marked phase, where the particles are located (when they leave this phase, they are suppressed). By default, for a the two phase fluide, the particles are supposed to be into the phase 0 (liquid).""",
        default=None,
    )
    methode_transport: Optional[Literal["vitesse_interpolee", "vitesse_particules"]] = (
        Field(
            description=r"""Kind of transport method for the particles. With vitesse_interpolee, the velocity of the particles is the velocity a fluid interpolation velocity (option by default). With vitesse_particules, the velocity of the particules is governed by the resolution of a momentum equation for the particles.""",
            default=None,
        )
    )
    methode_couplage: Optional[
        Literal["suivi", "one_way_coupling", "two_way_coupling"]
    ] = Field(
        description=r"""Way of coupling between the fluid and the particles. By default, (keyword suivi), there is no interaction between both. With one_way_coupling keyword, the fluid act on the particles. With two_way_coupling keyword, besides, particles act on the fluid.""",
        default=None,
    )
    nb_iterations: Optional[int] = Field(
        description=r"""Number of sub-timesteps to solve the momentum equation for the particles (1 per default).""",
        default=None,
    )
    contribution_one_way: Optional[Literal[0, 1]] = Field(
        description=r"""Activate (1, default) or not (0) the fluid forces on the particles when one_way_coupling or two_way_coupling coupling method is used.""",
        default=None,
    )
    implicite: Optional[Literal[0, 1]] = Field(
        description=r"""Impliciting (1) or not (0) the time scheme when weight added source term is used in the momentum equation""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "conditions_initiales": ["initial_conditions"],
        "injection": [],
        "transformation_bulles": [],
        "phase_marquee": [],
        "methode_transport": [],
        "methode_couplage": [],
        "nb_iterations": [],
        "contribution_one_way": [],
        "implicite": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Penalisation_forcage(Objet_lecture):
    r"""
    penalisation_forcage
    """

    pression_reference: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    domaine_flottant_fluide: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "pression_reference": [],
        "domaine_flottant_fluide": [],
    }


################################################################


class Traitement_particulier(Objet_lecture):
    r"""
    Auxiliary class to post-process particular values.
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    trait_part: Traitement_particulier_base = Field(
        description=r"""Type of traitement_particulier.""",
        default_factory=lambda: eval("Traitement_particulier_base()"),
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {None: ["nul"], "aco": [], "trait_part": [], "acof": []}


################################################################


class Navier_stokes_standard(Eqn_base):
    r"""
    Navier-Stokes equations.
    """

    correction_matrice_projection_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix matrix of initial projection for PDF""",
        default=None,
    )
    correction_calcul_pression_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix initial pressure computation for PDF""",
        default=None,
    )
    correction_vitesse_projection_initiale: Optional[int] = Field(
        description=r"""(IBM advanced) fix initial velocity computation for PDF""",
        default=None,
    )
    correction_matrice_pression: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure matrix for PDF""", default=None
    )
    correction_vitesse_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix velocity for PDF""", default=None
    )
    gradient_pression_qdm_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure gradient""", default=None
    )
    correction_pression_modifie: Optional[int] = Field(
        description=r"""(IBM advanced) fix pressure for PDF""", default=None
    )
    postraiter_gradient_pression_sans_masse: Optional[bool] = Field(
        description=r"""(IBM advanced) avoid mass matrix multiplication for the gradient postprocessing""",
        default=None,
    )
    solveur_pression: Optional[Solveur_sys_base] = Field(
        description=r"""Linear pressure system resolution method.""", default=None
    )
    dt_projection: Optional[Deuxmots] = Field(
        description=r"""nb value : This keyword checks every nb time-steps the equality of velocity divergence to zero. value is the criteria convergency for the solver used.""",
        default=None,
    )
    traitement_particulier: Optional[Traitement_particulier] = Field(
        description=r"""Keyword to post-process particular values.""", default=None
    )
    seuil_divu: Optional[Floatfloat] = Field(
        description=r"""value factor : this keyword is intended to minimise the number of iterations during the pressure system resolution. The convergence criteria during this step (\'seuil\' in solveur_pression) is dynamically adapted according to the mass conservation. At tn , the linear system Ax=B is considered as solved if the residual ||Ax-B||<seuil(tn). For tn+1, the threshold value seuil(tn+1) will be evualated as:  If ( |max(DivU)*dt|<value )  Seuil(tn+1)= Seuil(tn)*factor  Else  Seuil(tn+1)= Seuil(tn)*factor  Endif  The first parameter (value) is the mass evolution the user is ready to accept per timestep, and the second one (factor) is the factor of evolution for \'seuil\' (for example 1.1, so 10% per timestep). Investigations has to be lead to know more about the effects of these two last parameters on the behaviour of the simulations.""",
        default=None,
    )
    solveur_bar: Optional[Solveur_sys_base] = Field(
        description=r"""This keyword is used to define when filtering operation is called (typically for EF convective scheme, standard diffusion operator and Source_Qdm_lambdaup ). A file (solveur.bar) is then created and used for inversion procedure. Syntax is the same then for pressure solver (GCP is required for multi-processor calculations and, in a general way, for big meshes).""",
        default=None,
    )
    projection_initiale: Optional[int] = Field(
        description=r"""Keyword to suppress, if boolean equals 0, the initial projection which checks DivU=0. By default, boolean equals 1.""",
        default=None,
    )
    methode_calcul_pression_initiale: Optional[
        Literal[
            "avec_les_cl", "avec_sources", "avec_sources_et_operateurs", "sans_rien"
        ]
    ] = Field(
        description=r"""Keyword to select an option for the pressure calculation before the fist time step. Options are : avec_les_cl (default option lapP=0 is solved with Neuman boundary conditions on pressure if any), avec_sources (lapP=f is solved with Neuman boundaries conditions and f integrating the source terms of the Navier-Stokes equations) and avec_sources_et_operateurs (lapP=f is solved as with the previous option avec_sources but f integrating also some operators of the Navier-Stokes equations). The two last options are useful and sometime necessary when source terms are implicited when using an implicit time scheme to solve the Navier-Stokes equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Navier_stokes_turbulent(Navier_stokes_standard):
    r"""
    Navier-Stokes equations as well as the associated turbulence model equations.
    """

    modele_turbulence: Optional[Modele_turbulence_hyd_deriv] = Field(
        description=r"""Turbulence model for Navier-Stokes equations.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Navier_stokes_ft_disc(Navier_stokes_turbulent):
    r"""
    Two-phase momentum balance equation.
    """

    equation_interfaces_proprietes_fluide: Optional[str] = Field(
        description=r"""This keyword is used for liquid-gas, liquid-vapor and fluid-fluid deformable interface, which transported at the Eulerian velocity. When this case is selected, the keyword sequence Methode_transport vitesse_interpolee is used in the block Transport_Interfaces_FT_Disc to define the velocity field for the displacement of the interface.""",
        default=None,
    )
    equation_interfaces_vitesse_imposee: Optional[str] = Field(
        description=r"""This keyword is used to specify the velocity field to be used when using an interface that mimics a solid interface moving with a given solid speed of displacement. When this case is selected, the keyword sequence Methode_transport vitesse_imposee in the Transport_Interfaces_FT_Disc block will define the velocity field for the displacement of the interface.""",
        default=None,
    )
    equations_interfaces_vitesse_imposee: Optional[List[str]] = Field(
        description=r"""This keyword is used to specify the velocity field to be used when using an interface that mimics a solid interface moving with a given solid speed of displacement. When this case is selected, the keyword sequence Methode_transport vitesse_imposee in the Transport_Interfaces_FT_Disc block will define the velocity field for the displacement of the interface. If two or more solid interfaces are defined, then the keyword equations_interfaces_vitesse_imposee should be used.""",
        default=None,
    )
    clipping_courbure_interface: Optional[int] = Field(
        description=r"""This keyword is used to numerically limit the values of curvature used in the momentum balance equation. Curvature is computed as usual, but values exceeding the clipping value are replaced by this threshold, before using the clipped curvature in the momentum balance. Each time a curvature value is clipped, a counter is increased by one unity and the value of the counter is written in the .err file at the end of the time step. This clipping allows not reducing drastically the time stepping when a geometrical singularity occurs in the interface mesh. However, physical phenomena may be concealed with the use of such a clipping.""",
        default=None,
    )
    terme_gravite: Optional[Literal["rho_g", "grad_i"]] = Field(
        description=r"""The Terme_gravite keyword changes the numerical scheme used for the gravity source term. The default is grad_i, which is designed to remove spurious currents around the interface. In this case, the pressure field does not contain the hydrostatic part but only a jump across the interface. This scheme seems not to work very well in vef. The rho_g option uses the more traditional source term, equal to rho*g in the volume. In this case, the hydrostatic pressure is visible in the pressure field and the boundary conditions in pressure must be set accordingly. This model produces spurious currents in the vicinity of the fluid-fluid interfaces and with the immersed boundary conditions.""",
        default=None,
    )
    equation_temperature_mpoint: Optional[str] = Field(
        description=r"""The equation_temperature_mpoint should be used in the case of liquid-vapor flow with phase-change (see the TRUST_ROOT/doc/TRUST/ft_chgt_phase.pdf written in French for more information about the model). The name of the temperature equation, defined with the convection_diffusion_temperature_ft_disc keyword, should be given.""",
        default=None,
    )
    matrice_pression_invariante: Optional[bool] = Field(
        description=r"""This keyword is a shortcut to be used only when the flow is a single-phase one, with interface tracking only used for solid-fluid interfaces. In this peculiar case, the density of the fluid does not evolve during the computation and the pressure matrix does not need to be actuated at each time step.""",
        default=None,
    )
    penalisation_forcage: Optional[Penalisation_forcage] = Field(
        description=r"""This keyword is used to specify a strong formulation (value set to 0) or a weak formulation (value set to 1) for an imposed pressure boundary condition. The first formulation converges quicker and is stable in general cases except some rare cases (see Ecoulement_Neumann test case for example) where the second one should be used despite of its slow convergence.""",
        default=None,
    )
    equation_temperature_mpoint_vapeur: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    mpoint_inactif_sur_qdm: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    mpoint_vapeur_inactif_sur_qdm: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    new_mass_source: Optional[bool] = Field(
        description=r"""Flag for localised computation of velocity jump based on interfacial area AI (advanced option)""",
        default=None,
    )
    interpol_indic_pour_di_dt: Optional[
        Literal["interp_ai_based", "interp_standard", "interp_modifiee"]
    ] = Field(
        description=r"""Specific interpolation of phase indicator function in VoF mass-preserving method (advanced option)""",
        default=None,
    )
    outletcorrection_pour_di_dt: Optional[Literal["correction_ghost_indic"]] = Field(
        description=r"""not_set""", default=None
    )
    boussinesq_approximation: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "equation_interfaces_proprietes_fluide": [],
        "equation_interfaces_vitesse_imposee": [],
        "equations_interfaces_vitesse_imposee": [],
        "clipping_courbure_interface": [],
        "terme_gravite": [],
        "equation_temperature_mpoint": [],
        "matrice_pression_invariante": [],
        "penalisation_forcage": [],
        "equation_temperature_mpoint_vapeur": [],
        "mpoint_inactif_sur_qdm": [],
        "mpoint_vapeur_inactif_sur_qdm": [],
        "new_mass_source": [],
        "interpol_indic_pour_di_dt": [],
        "outletcorrection_pour_di_dt": [],
        "boussinesq_approximation": [],
        "modele_turbulence": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Source_trainee(Source_base):
    r"""
    drag effect
    """

    _synonyms: ClassVar[dict] = {None: ["trainee"]}


################################################################


class Source_flottabilite(Source_base):
    r"""
    buoyancy effect
    """

    _synonyms: ClassVar[dict] = {None: ["flottabilite"]}


################################################################


class Source_masse_ajoutee(Source_base):
    r"""
    weight added effect
    """

    _synonyms: ClassVar[dict] = {None: ["masse_ajoutee"]}


################################################################


class Pb_gen_base(Objet_u):
    r"""
    Basic class for problems.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Un_pb(Objet_lecture):
    r"""
    pour les groupes
    """

    mot: str = Field(description=r"""the string""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": []}


################################################################


class Coupled_problem(Pb_gen_base):
    r"""
    This instruction causes a probleme_couple type object to be created. This type of object
    has an associated problem list, that is, the coupling of n problems among them may be
    processed. Coupling between these problems is carried out explicitly via conditions at
    particular contact limits. Each problem may be associated either with the Associate
    keyword or with the Read/groupes keywords. The difference is that in the first case, the
    four problems exchange values then calculate their timestep, rather in the second case,
    the same strategy is used for all the problems listed inside one group, but the second
    group of problem exchange values with the first group of problems after the first group
    did its timestep. So, the first case may then also be written like this:

    Probleme_Couple pbc

    Read pbc { groupes { { pb1 , pb2 , pb3 , pb4 } } }

    There is a physical environment per problem (however, the same physical environment could
    be common to several problems).

    Each problem is resolved in a domain.

    Warning : Presently, coupling requires coincident meshes. In case of non-coincident
    meshes, boundary condition \'paroi_contact\' in VEF returns error message (see
    paroi_contact for correcting procedure).
    """

    groupes: Optional[
        Annotated[List[Annotated[List[Un_pb], "List_un_pb"]], "List_list_nom"]
    ] = Field(description=r"""pour les groupes""", default=None)
    _synonyms: ClassVar[dict] = {None: ["probleme_couple"], "groupes": []}


################################################################


class Probleme_couple_rayonnement(Coupled_problem):
    r"""
    This keyword is used to define a problem coupling several other problems to which
    radiation coupling is added.
    """

    _synonyms: ClassVar[dict] = {None: ["pb_couple_rayonnement"], "groupes": []}


################################################################


class Frontiere_ouverte_temperature_imposee(Dirichlet):
    r"""
    Imposed temperature condition at the open boundary called bord (edge) (in the case of
    fluid inlet). This condition must be associated with an imposed inlet velocity condition.
    The imposed temperature value is expressed in oC or K.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["frontiere_ouverte_enthalpie_imposee"],
        "ch": [],
    }


################################################################


class Frontiere_ouverte_temperature_imposee_rayo_transp(
    Frontiere_ouverte_temperature_imposee
):
    r"""
    Imposed temperature condition for a radiation problem with transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_echange_externe_impose(Condlim_base):
    r"""
    External type exchange condition with a heat exchange coefficient and an imposed external
    temperature.
    """

    h_or_t: Literal["h_imp", "t_ext"] = Field(
        description=r"""Heat exchange coefficient value (expressed in W.m-2.K-1).""",
        default="h_imp",
    )
    himpc: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    t_or_h: Literal["t_ext", "h_imp"] = Field(
        description=r"""External temperature value (expressed in oC or K).""",
        default="t_ext",
    )
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Paroi_echange_externe_impose_rayo_transp(Paroi_echange_externe_impose):
    r"""
    External type exchange condition for a coupled problem with radiation in transparent gas.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Modele_rayonnement_base(Objet_u):
    r"""
    Basic class for wall thermal radiation model.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Modele_rayonnement_milieu_transparent(Modele_rayonnement_base):
    r"""
    Wall thermal radiation model for a transparent gas and resolving a radiation-conduction-
    thermohydraulics coupled problem in VDF or VEF.
    """

    bloc: Bloc_lecture = Field(
        description=r"""Modele_Rayonnement_Milieu_Transparent mod  Read mod {  nom_pb_rayonnant  problem_name  fichier_fij  file_name  fichier_face_rayo  file_name  [fichier_matrice | fichier_matrice_binaire file_name]  }   nom_pb_rayonnant problem_name : problem_name is the name of the radiating fluid problem  fichier_fij file_name : file_name is the name of the file which contains the shape factor matrix between all the faces.  fichier_face_rayo file_name : file_name is the name of the file which contains the radiating faces characteristics (area, emission value ...)  fichier_matrice|fichier_matrice_binaire file_name : file_name is the name of the ASCII (or binary) file which contains the inverted shape factor matrix. It is an optional keyword, if not defined, the inverted shape factor matrix will be calculated and written in a file.  The two first files can be generated by a preprocessor, they allow the radiating face characteristics to be entered (set of faces considered to be uniform with respect to radiation for emission value, flux, etc.) and the form factors for these various faces. These files have the following format:  File on radiating faces:  N M			-> N is the number of radiating faces (=edges) and M equals the number of non-zero emission radiating faces  Nom(i) S(i) E(i)	-> Name of the edge i, surface area of the edge i -> emission value (between 0 an 1)  Exemple:  13 4  Gauche 50.0 0.0  Droit1 50.0 0.5  Bas 10.0 0.0  Haut 10.0 0.0  Arriere 5.0 0.0  Avant 5.0 0.0  Droit2 30.0 0.5  Bas1 40.0 0.0  Haut1 20.0 0.0  Avant1 20.0 0.0  Arriere1 20.0 0.0  Entree 20.0 0.5  Sortie 20.0 0.5   File on form factors:  N -> Number of radiating faces  Fij -> Matrix of form factors where i, j between 1 and N   Example:  13  1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  0.00 0.00 0.00 0.00 0.00 0.00 0.24 0.20 0.10 0.10 0.10 0.10 0.16  0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  0.00 0.40 0.00 0.00 0.00 0.00 0.00 0.20 0.10 0.10 0.10 0.10 0.00  0.00 0.25 0.00 0.00 0.00 0.00 0.15 0.00 0.15 0.10 0.10 0.15 0.10  0.00 0.25 0.00 0.00 0.00 0.00 0.15 0.30 0.00 0.10 0.10 0.00 0.10  0.00 0.25 0.00 0.00 0.00 0.00 0.15 0.20 0.10 0.00 0.10 0.10 0.10  0.00 0.25 0.00 0.00 0.00 0.00 0.15 0.20 0.10 0.10 0.00 0.10 0.10  0.00 0.25 0.00 0.00 0.00 0.00 0.15 0.30 0.00 0.10 0.10 0.00 0.10  0.00 0.40 0.00 0.00 0.00 0.00 0.00 0.20 0.10 0.10 0.10 0.10 0.00   Caution:  a) The radiation model\'s precision is decided by the user when he/she names the domain edges. In fact, a radiating face is recognised by the preprocessor as the set of domain edges faces bearing the same name. Thus, if the user subdivides the edge into two edges which are named differently, he/she thus creates two radiating faces instead of one.  b) The form factors are entered by the user, the preprocessor carries out no calculations other than checking preservation relationships on form factors.  c) The fluid is considered to be a transparent gas.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Paroi_flux_impose(Condlim_base):
    r"""
    Normal flux condition at the wall called bord (edge). The surface area of the flux (W.m-1
    in 2D or W.m-2 in 3D) is imposed at the boundary according to the following convention: a
    positive flux is a flux that enters into the domain according to convention.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_flux_impose_rayo_transp(Paroi_flux_impose):
    r"""
    Normal flux condition at the wall called bord (edge) for a radiation problem in
    transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_temperature_imposee(Dirichlet):
    r"""
    Imposed temperature condition at the wall called bord (edge).
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_temperature_imposee_rayo_transp(Paroi_temperature_imposee):
    r"""
    Imposed temperature condition at the wall called bord (edge) for a radiation problem in
    transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Champ_front_contact_vef(Front_field_base):
    r"""
    This field is used on a boundary between a solid and fluid domain to exchange a calculated
    temperature at the contact face of the two domains according to the flux of the two
    problems.
    """

    local_pb: str = Field(description=r"""Name of the problem.""", default="")
    local_boundary: str = Field(description=r"""Name of the boundary.""", default="")
    remote_pb: str = Field(description=r"""Name of the second problem.""", default="")
    remote_boundary: str = Field(
        description=r"""Name of the boundary in the second problem.""", default=""
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "local_pb": [],
        "local_boundary": [],
        "remote_pb": [],
        "remote_boundary": [],
    }


################################################################


class Champ_front_contact_rayo_transp_vef(Champ_front_contact_vef):
    r"""
    This field is used on a boundary between a solid and fluid domain to exchange a calculated
    temperature at the contact face of the two domains according to the flux of the two
    problems with radiation in transparent fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "local_pb": [],
        "local_boundary": [],
        "remote_pb": [],
        "remote_boundary": [],
    }


################################################################


class Echange_contact_rayo_transp_vdf(Paroi_echange_contact_vdf):
    r"""
    Exchange boundary condition in VDF between the transparent fluid and the solid for a
    problem coupled with radiation. Without radiation, it is the equivalent of the
    Paroi_Echange_contact_VDF exchange condition.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "temp": [],
        "h": [],
    }


################################################################


class Neumann(Condlim_base):
    r"""
    Neumann condition at the boundary called bord (edge) : 1). For Navier-Stokes equations,
    constraint imposed at the boundary; 2). For scalar transport equation, flux imposed at the
    boundary.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Frontiere_ouverte(Neumann):
    r"""
    Boundary outlet condition on the boundary called bord (edge) (diffusion flux zero). This
    condition must be associated with a boundary outlet hydraulic condition.
    """

    var_name: Literal[
        "t_ext",
        "c_ext",
        "y_ext",
        "k_eps_ext",
        "k_omega_ext",
        "fluctu_temperature_ext",
        "flux_chaleur_turb_ext",
        "v2_ext",
        "a_ext",
        "tau_ext",
        "k_ext",
        "omega_ext",
        "h_ext",
    ] = Field(description=r"""Field name.""", default="t_ext")
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "var_name": [], "ch": []}


################################################################


class Frontiere_ouverte_rayo_transp(Frontiere_ouverte):
    r"""
    Keyword to set a boundary outlet temperature condition on the boundary called bord (edge)
    (diffusion flux zero) for a radiation problem with transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "var_name": [], "ch": []}


################################################################


class Frontiere_ouverte_rayo_transp_vdf(Frontiere_ouverte_rayo_transp):
    r"""
    doit disparaitre
    """

    _synonyms: ClassVar[dict] = {None: [], "var_name": [], "ch": []}


################################################################


class Frontiere_ouverte_rayo_transp_vef(Frontiere_ouverte_rayo_transp):
    r"""
    doit disparaitre
    """

    _synonyms: ClassVar[dict] = {None: [], "var_name": [], "ch": []}


################################################################


class Frontiere_ouverte_temperature_imposee_rayo_semi_transp(
    Frontiere_ouverte_temperature_imposee
):
    r"""
    Imposed temperature condition for a radiation problem with semi transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_echange_externe_impose_rayo_semi_transp(Paroi_echange_externe_impose):
    r"""
    External type exchange condition for a coupled problem with radiation in semi transparent
    gas.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Paroi_flux_impose_rayo_semi_transp_vdf(Paroi_flux_impose):
    r"""
    Normal flux condition at the wall called bord (edge) for a radiation problem in semi
    transparent gas (in VDF).
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_flux_impose_rayo_semi_transp_vef(Paroi_flux_impose):
    r"""
    Normal flux condition at the wall called bord (edge) for a radiation problem in semi
    transparent gas (in VEF).
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_temperature_imposee_rayo_semi_transp(Paroi_temperature_imposee):
    r"""
    Imposed temperature condition at the wall called bord (edge) for a radiation problem in
    semi transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Champ_front_contact_rayo_semi_transp_vef(Champ_front_contact_vef):
    r"""
    This field is used on a boundary between a solid and fluid domain to exchange a calculated
    temperature at the contact face of the two domains according to the flux of the two
    problems with radiation in semi transparent fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "local_pb": [],
        "local_boundary": [],
        "remote_pb": [],
        "remote_boundary": [],
    }


################################################################


class Paroi_echange_contact_rayo_semi_transp_vdf(Paroi_echange_contact_vdf):
    r"""
    Exchange boundary condition in VDF between the semi transparent fluid and the solid for a
    problem coupled with radiation.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "temp": [],
        "h": [],
    }


################################################################


class Frontiere_ouverte_rayo_semi_transp(Frontiere_ouverte):
    r"""
    Keyword to set a boundary outlet temperature condition on the boundary called bord (edge)
    (diffusion flux zero) for a radiation problem with semi transparent gas.
    """

    _synonyms: ClassVar[dict] = {None: [], "var_name": [], "ch": []}


################################################################


class Eq_rayo_semi_transp(Objet_lecture):
    r"""
    Irradiancy equation.
    """

    solveur: Solveur_sys_base = Field(
        description=r"""Solver of the irradiancy equation.""",
        default_factory=lambda: eval("Solveur_sys_base()"),
    )
    conditions_limites: Optional[Annotated[List[Condlimlu], "Condlims"]] = Field(
        description=r"""Boundary conditions.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "solveur": [],
        "conditions_limites": ["boundary_conditions"],
    }


################################################################


class Bloc_lecture_poro(Objet_lecture):
    r"""
    Surface and volume porosity values.
    """

    volumique: float = Field(description=r"""Volume porosity value.""", default=0.0)
    surfacique: List[float] = Field(
        description=r"""Surface porosity values (in X, Y, Z directions).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "volumique": [], "surfacique": []}


################################################################


class Porosites(Objet_u):
    r"""
    To define the volume porosity and surface porosity that are uniform in every direction in
    space on a sub-area.

    Porosity was only usable in VDF discretization, and now available for VEF P1NC/P0.

    Observations :

    - Surface porosity values must be given in every direction in space (set this value to 1
    if there is no porosity),

    - Prior to defining porosity, the problem must have been discretized.

    Can \'t be used in VEF discretization, use Porosites_champ instead.
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    sous_zone: str = Field(
        description=r"""Name of the sub-area to which porosity are allocated.""",
        default="",
    )
    bloc: Bloc_lecture_poro = Field(
        description=r"""Surface and volume porosity values.""",
        default_factory=lambda: eval("Bloc_lecture_poro()"),
    )
    sous_zone2: Optional[str] = Field(
        description=r"""Name of the 2nd sub-area to which porosity are allocated.""",
        default=None,
    )
    bloc2: Optional[Bloc_lecture_poro] = Field(
        description=r"""Surface and volume porosity values.""", default=None
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: [],
        "aco": [],
        "sous_zone": ["sous_zone1"],
        "bloc": [],
        "sous_zone2": [],
        "bloc2": [],
        "acof": [],
    }


################################################################


class Milieu_base(Objet_u):
    r"""
    Basic class for medium (physics properties of medium).
    """

    gravite: Optional[Field_base] = Field(
        description=r"""Gravity field (optional).""", default=None
    )
    porosites_champ: Optional[Field_base] = Field(
        description=r"""The porosity is given at each element and the porosity at each face, Psi(face), is calculated by the average of the porosities of the two neighbour elements Psi(elem1), Psi(elem2) : Psi(face)=2/(1/Psi(elem1)+1/Psi(elem2)). This keyword is optional.""",
        default=None,
    )
    diametre_hyd_champ: Optional[Field_base] = Field(
        description=r"""Hydraulic diameter field (optional).""", default=None
    )
    porosites: Optional[Porosites] = Field(description=r"""Porosities.""", default=None)
    rho: Optional[Field_base] = Field(
        description=r"""Density (kg.m-3).""", default=None
    )
    lambda_: Optional[Field_base] = Field(
        description=r"""Conductivity (W.m-1.K-1).""", default=None
    )
    cp: Optional[Field_base] = Field(
        description=r"""Specific heat (J.kg-1.K-1).""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Constituant(Milieu_base):
    r"""
    Constituent.
    """

    coefficient_diffusion: Optional[Field_base] = Field(
        description=r"""Constituent diffusion coefficient value (m2.s-1). If a multi-constituent problem is being processed, the diffusivite will be a vectorial and each components will be the diffusion of the constituent.""",
        default=None,
    )
    is_multi_scalar: Optional[bool] = Field(
        description=r"""Flag to activate the multi_scalar diffusion operator""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "coefficient_diffusion": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Definition_champs(Listobj):
    r"""
    List of definition champ
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Definition_champs_fichier(Objet_lecture):
    r"""
    Keyword to read definition_champs from a file
    """

    fichier: str = Field(description=r"""name of file""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "fichier": ["file"]}


################################################################


class Sondes(Listobj):
    r"""
    List of probes.
    """

    _synonyms: ClassVar[dict] = {None: ["nul", "probes"]}


################################################################


class Sondes_fichier(Objet_lecture):
    r"""
    Keyword to read probes from a file
    """

    fichier: str = Field(description=r"""name of file""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "fichier": ["file"]}


################################################################


class Champs_a_post(Listobj):
    r"""
    Fields to be post-processed.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Champ_a_post(Objet_lecture):
    r"""
    Field to be post-processed.
    """

    champ: str = Field(description=r"""Name of the post-processed field.""", default="")
    localisation: Optional[Literal["elem", "som", "faces"]] = Field(
        description=r"""Localisation of post-processed field values: The two available values are elem, som, or faces (LATA format only) used respectively to select field values at mesh centres (CHAMPMAILLE type field in the lml file) or at mesh nodes (CHAMPPOINT type field in the lml file). If no selection is made, localisation is set to som by default.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "champ": [], "localisation": []}


################################################################


class Champs_posts(Objet_lecture):
    r"""
    Field\'s write mode.
    """

    format: Optional[Literal["binaire", "formatte"]] = Field(
        description=r"""Type of file.""", default=None
    )
    mot: Literal["dt_post", "nb_pas_dt_post"] = Field(
        description=r"""Keyword to set the kind of the field\'s write frequency. Either a time period or a time step period.""",
        default="dt_post",
    )
    period: str = Field(
        description=r"""Value of the period which can be like (2.*t).""", default=""
    )
    champs: Annotated[List[Champ_a_post], "Champs_a_post"] = Field(
        description=r"""Fields to be post-processed.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "format": [],
        "mot": [],
        "period": [],
        "champs": ["fields"],
    }


################################################################


class Champs_posts_fichier(Objet_lecture):
    r"""
    Fields read from file.
    """

    format: Optional[Literal["binaire", "formatte"]] = Field(
        description=r"""Type of file.""", default=None
    )
    mot: Optional[Literal["dt_post", "nb_pas_dt_post"]] = Field(
        description=r"""Keyword to set the kind of the field\'s write frequency. Either a time period or a time step period.""",
        default=None,
    )
    period: Optional[str] = Field(
        description=r"""Value of the period which can be like (2.*t).""", default=None
    )
    fichier: str = Field(description=r"""name of file""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "format": [],
        "mot": [],
        "period": [],
        "fichier": ["file"],
    }


################################################################


class List_stat_post(Listobj):
    r"""
    Post-processing for statistics
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Stat_post_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Stats_posts(Objet_lecture):
    r"""
    Post-processing for statistics. \input{{statistiques}}
    """

    mot: Optional[Literal["dt_post", "nb_pas_dt_post"]] = Field(
        description=r"""Keyword to set the kind of the field\'s write frequency. Either a time period or a time step period.""",
        default=None,
    )
    period: Optional[str] = Field(
        description=r"""Value of the period which can be like (2.*t).""", default=None
    )
    champs: Annotated[List[Stat_post_deriv], "List_stat_post"] = Field(
        description=r"""Post-processing for statistics""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "mot": [],
        "period": [],
        "champs": ["fields"],
    }


################################################################


class Stats_posts_fichier(Objet_lecture):
    r"""
    Statistics read from file.. \input{{statistiques}}
    """

    mot: Literal["dt_post", "nb_pas_dt_post"] = Field(
        description=r"""Keyword to set the kind of the field\'s write frequency. Either a time period or a time step period.""",
        default="dt_post",
    )
    period: str = Field(
        description=r"""Value of the period which can be like (2.*t).""", default=""
    )
    fichier: str = Field(description=r"""name of file""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "mot": [],
        "period": [],
        "fichier": ["file"],
    }


################################################################


class Stats_serie_posts(Objet_lecture):
    r"""
    This keyword is used to set the statistics. Average on dt_integr time interval is post-
    processed every dt_integr seconds. \input{{statistiquesseries}}
    """

    mot: Literal["dt_integr"] = Field(
        description=r"""Keyword is used to set the statistics period of integration and write period.""",
        default="dt_integr",
    )
    dt_integr: float = Field(
        description=r"""Average on dt_integr time interval is post-processed every dt_integr seconds.""",
        default=0.0,
    )
    stat: Annotated[List[Stat_post_deriv], "List_stat_post"] = Field(
        description=r"""Post-processing for statistics""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": [], "dt_integr": [], "stat": []}


################################################################


class Stats_serie_posts_fichier(Objet_lecture):
    r"""
    This keyword is used to set the statistics read from a file. Average on dt_integr time
    interval is post-processed every dt_integr seconds. \input{{statistiquesseries}}
    """

    mot: Literal["dt_integr"] = Field(
        description=r"""Keyword is used to set the statistics period of integration and write period.""",
        default="dt_integr",
    )
    dt_integr: float = Field(
        description=r"""Average on dt_integr time interval is post-processed every dt_integr seconds.""",
        default=0.0,
    )
    fichier: str = Field(description=r"""name of file""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "mot": [],
        "dt_integr": [],
        "fichier": ["file"],
    }


################################################################


class Champ_generique_base(Objet_u):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Definition_champ(Objet_lecture):
    r"""
    Keyword to create new complex field for advanced postprocessing.
    """

    name: str = Field(description=r"""The name of the new created field.""", default="")
    champ_generique: Champ_generique_base = Field(
        description=r"""not_set""",
        default_factory=lambda: eval("Champ_generique_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "name": [], "champ_generique": []}


################################################################


class Sonde_base(Objet_lecture):
    r"""
    Basic probe. Probes refer to sensors that allow a value or several points of the domain to
    be monitored over time. The probes may be a set of points defined one by one (keyword
    Points) or a set of points evenly distributed over a straight segment (keyword Segment) or
    arranged according to a layout (keyword Plan) or according to a parallelepiped (keyword
    Volume). The fields allow all the values of a physical value on the domain to be known at
    several moments in time.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Sonde(Objet_lecture):
    r"""
    Keyword is used to define the probes. Observations: the probe coordinates should be given
    in Cartesian coordinates (X, Y, Z), including axisymmetric.
    """

    nom_sonde: str = Field(
        description=r"""Name of the file in which the values taken over time will be saved. The complete file name is nom_sonde.son.""",
        default="",
    )
    special: Optional[Literal["grav", "som", "nodes", "chsom", "gravcl"]] = Field(
        description=r"""Option to change the positions of the probes. Several options are available:  grav : each probe is moved to the nearest cell center of the mesh;  som : each probe is moved to the nearest vertex of the mesh  nodes : each probe is moved to the nearest face center of the mesh;  chsom : only available for P1NC sampled field. The values of the probes are calculated according to P1-Conform corresponding field.  gravcl : Extend to the domain face boundary a cell-located segment probe in order to have the boundary condition for the field. For this type the extreme probe point has to be on the face center of gravity.""",
        default=None,
    )
    nom_inco: str = Field(description=r"""Name of the sampled field.""", default="")
    mperiode: Literal["periode"] = Field(
        description=r"""Keyword to set the sampled field measurement frequency.""",
        default="periode",
    )
    prd: float = Field(
        description=r"""Period value. Every prd seconds, the field value calculated at the previous time step is written to the nom_sonde.son file.""",
        default=0.0,
    )
    type: Sonde_base = Field(
        description=r"""Type of probe.""", default_factory=lambda: eval("Sonde_base()")
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "nom_sonde": [],
        "special": [],
        "nom_inco": [],
        "mperiode": [],
        "prd": [],
        "type": [],
    }


################################################################


class Postraitement(Postraitement_base):
    r"""
    An object of post-processing (without name).
    """

    fichier: Optional[str] = Field(description=r"""Name of file.""", default=None)
    format: Optional[
        Literal["lml", "lata", "single_lata", "lata_v2", "med", "med_major", "cgns"]
    ] = Field(
        description=r"""This optional parameter specifies the format of the output file. The basename used for the output file is the basename of the data file. For the fmt parameter, choices are lml or lata. A short description of each format can be found below. The default value is lml.""",
        default=None,
    )
    dt_post: Optional[int] = Field(
        description=r"""Field\'s write frequency (as a time period) - can also be specified after the 'field' keyword.""",
        default=None,
    )
    nb_pas_dt_post: Optional[int] = Field(
        description=r"""Field\'s write frequency (as a number of time steps) - can also be specified after the 'field' keyword.""",
        default=None,
    )
    domaine: Optional[str] = Field(
        description=r"""This optional parameter specifies the domain on which the data should be interpolated before it is written in the output file. The default is to write the data on the domain of the current problem (no interpolation).""",
        default=None,
    )
    sous_domaine: Optional[str] = Field(
        description=r"""This optional parameter specifies the sub_domaine on which the data should be interpolated before it is written in the output file. It is only available for sequential computation.""",
        default=None,
    )
    parallele: Optional[Literal["simple", "multiple", "mpi-io"]] = Field(
        description=r"""Select simple (single file, sequential write), multiple (several files, parallel write), or mpi-io (single file, parallel write) for LATA format""",
        default=None,
    )
    definition_champs: Optional[
        Annotated[List[Definition_champ], "Definition_champs"]
    ] = Field(description=r"""List of definition champ""", default=None)
    definition_champs_fichier: Optional[Definition_champs_fichier] = Field(
        description=r"""Definition_champs read from file.""", default=None
    )
    sondes: Optional[Annotated[List[Sonde], "Sondes"]] = Field(
        description=r"""List of probes.""", default=None
    )
    sondes_fichier: Optional[Sondes_fichier] = Field(
        description=r"""Probe read from a file.""", default=None
    )
    sondes_mobiles: Optional[Annotated[List[Sonde], "Sondes"]] = Field(
        description=r"""List of probes.""", default=None
    )
    sondes_mobiles_fichier: Optional[Sondes_fichier] = Field(
        description=r"""Mobile probes read in a file""", default=None
    )
    deprecatedkeepduplicatedprobes: Optional[int] = Field(
        description=r"""Flag to not remove duplicated probes in .son files (1: keep duplicate probes, 0: remove duplicate probes)""",
        default=None,
    )
    champs: Optional[Champs_posts] = Field(
        description=r"""Field\'s write mode.""", default=None
    )
    champs_fichier: Optional[Champs_posts_fichier] = Field(
        description=r"""Fields read from file.""", default=None
    )
    statistiques: Optional[Stats_posts] = Field(
        description=r"""Statistics between two points fixed : start of integration time and end of integration time.""",
        default=None,
    )
    statistiques_fichier: Optional[Stats_posts_fichier] = Field(
        description=r"""Statistics read from file.""", default=None
    )
    statistiques_en_serie: Optional[Stats_serie_posts] = Field(
        description=r"""Statistics between two points not fixed : on period of integration.""",
        default=None,
    )
    statistiques_en_serie_fichier: Optional[Stats_serie_posts_fichier] = Field(
        description=r"""Serial_statistics read from a file""", default=None
    )
    suffix_for_reset: Optional[str] = Field(
        description=r"""Suffix used to modify the postprocessing file name if the ICoCo resetTime() method is invoked.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["post_processing"],
        "fichier": [],
        "format": [],
        "dt_post": [],
        "nb_pas_dt_post": [],
        "domaine": [],
        "sous_domaine": ["sous_zone"],
        "parallele": [],
        "definition_champs": [],
        "definition_champs_fichier": ["definition_champs_file"],
        "sondes": ["probes"],
        "sondes_fichier": ["probes_file"],
        "sondes_mobiles": ["mobile_probes"],
        "sondes_mobiles_fichier": ["mobile_probes_file"],
        "deprecatedkeepduplicatedprobes": [],
        "champs": ["fields"],
        "champs_fichier": ["fields_file"],
        "statistiques": ["statistics"],
        "statistiques_fichier": ["statistics_file"],
        "statistiques_en_serie": ["serial_statistics"],
        "statistiques_en_serie_fichier": ["serial_statistics_file"],
        "suffix_for_reset": [],
    }


################################################################


class Corps_postraitement(Postraitement):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "fichier": [],
        "format": [],
        "dt_post": [],
        "nb_pas_dt_post": [],
        "domaine": [],
        "sous_domaine": ["sous_zone"],
        "parallele": [],
        "definition_champs": [],
        "definition_champs_fichier": ["definition_champs_file"],
        "sondes": ["probes"],
        "sondes_fichier": ["probes_file"],
        "sondes_mobiles": ["mobile_probes"],
        "sondes_mobiles_fichier": ["mobile_probes_file"],
        "deprecatedkeepduplicatedprobes": [],
        "champs": ["fields"],
        "champs_fichier": ["fields_file"],
        "statistiques": ["statistics"],
        "statistiques_fichier": ["statistics_file"],
        "statistiques_en_serie": ["serial_statistics"],
        "statistiques_en_serie_fichier": ["serial_statistics_file"],
        "suffix_for_reset": [],
    }


################################################################


class Postraitements(Listobj):
    r"""
    Keyword to use several results files. List of objects of post-processing (with name).
    """

    _synonyms: ClassVar[dict] = {None: ["post_processings"]}


################################################################


class Liste_post_ok(Listobj):
    r"""
    Keyword to use several results files. List of objects of post-processing (with name)
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Liste_post(Listobj):
    r"""
    Keyword to use several results files. List of objects of post-processing (with name)
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Un_postraitement(Objet_lecture):
    r"""
    An object of post-processing (with name).
    """

    nom: str = Field(description=r"""Name of the post-processing.""", default="")
    post: Corps_postraitement = Field(
        description=r"""Definition of the post-processing.""",
        default_factory=lambda: eval("Corps_postraitement()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "nom": [], "post": []}


################################################################


class Nom_postraitement(Objet_lecture):
    r"""
    not_set
    """

    nom: str = Field(description=r"""Name of the post-processing.""", default="")
    post: Postraitement_base = Field(
        description=r"""the post""",
        default_factory=lambda: eval("Postraitement_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "nom": [], "post": []}


################################################################


class Type_un_post(Objet_lecture):
    r"""
    not_set
    """

    type: Literal["postraitement", "post_processing"] = Field(
        description=r"""not_set""", default="postraitement"
    )
    post: Un_postraitement = Field(
        description=r"""not_set""", default_factory=lambda: eval("Un_postraitement()")
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "type": [], "post": []}


################################################################


class Type_postraitement_ft_lata(Objet_lecture):
    r"""
    not_set
    """

    type: Literal["postraitement_ft_lata", "postraitement_lata"] = Field(
        description=r"""not_set""", default="postraitement_ft_lata"
    )
    nom: str = Field(description=r"""Name of the post-processing.""", default="")
    bloc: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: ["nul"], "type": [], "nom": [], "bloc": []}


################################################################


class Un_postraitement_spec(Objet_lecture):
    r"""
    An object of post-processing (with type +name).
    """

    type_un_post: Optional[Type_un_post] = Field(
        description=r"""not_set""", default=None
    )
    type_postraitement_ft_lata: Optional[Type_postraitement_ft_lata] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "type_un_post": [],
        "type_postraitement_ft_lata": [],
    }


################################################################


class Pb_base(Pb_gen_base):
    r"""
    Resolution of equations on a domain. A problem is defined by creating an object and
    assigning the problem type that the user wishes to resolve. To enter values for the
    problem objects created, the Lire (Read) interpretor is used with a data block.
    """

    milieu: Optional[Milieu_base] = Field(
        description=r"""The medium associated with the problem.""", default=None
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituent.""", default=None
    )
    postraitement: Optional[Corps_postraitement] = Field(
        description=r"""One post-processing (without name).""", default=None
    )
    postraitements: Optional[Annotated[List[Un_postraitement], "Postraitements"]] = (
        Field(
            description=r"""Keyword to use several results files. List of objects of post-processing (with name).""",
            default=None,
        )
    )
    liste_de_postraitements: Optional[
        Annotated[List[Nom_postraitement], "Liste_post_ok"]
    ] = Field(
        description=r"""Keyword to use several results files. List of objects of post-processing (with name)""",
        default=None,
    )
    liste_postraitements: Optional[
        Annotated[List[Un_postraitement_spec], "Liste_post"]
    ] = Field(
        description=r"""Keyword to use several results files. List of objects of post-processing (with name)""",
        default=None,
    )
    sauvegarde: Optional[Format_file] = Field(
        description=r"""Keyword used when calculation results are to be backed up. When a coupling is performed, the backup-recovery file name must be well specified for each problem. In this case, you must save to different files and correctly specify these files when resuming the calculation.""",
        default=None,
    )
    sauvegarde_simple: Optional[Format_file] = Field(
        description=r"""The same keyword than Sauvegarde except, the last time step only is saved.""",
        default=None,
    )
    reprise: Optional[Format_file] = Field(
        description=r"""Keyword to resume a calculation based on the name_file file (see the class format_file). If format_reprise is xyz, the name_file file should be the .xyz file created by the previous calculation. With this file, it is possible to resume a parallel calculation on P processors, whereas the previous calculation has been run on N (N<>P) processors. Should the calculation be resumed, values for the tinit (see schema_temps_base) time fields are taken from the name_file file. If there is no backup corresponding to this time in the name_file, TRUST exits in error.""",
        default=None,
    )
    resume_last_time: Optional[Format_file] = Field(
        description=r"""Keyword to resume a calculation based on the name_file file, resume the calculation at the last time found in the file (tinit is set to last time of saved files).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Modele_rayo_semi_transp(Pb_base):
    r"""
    Radiation model for semi transparent gas. The model should be associated to the coupling
    problem BEFORE the time scheme.
    """

    eq_rayo_semi_transp: Optional[Eq_rayo_semi_transp] = Field(
        description=r"""Irradiancy G equation. Radiative flux equals -grad(G)/3/kappa.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "eq_rayo_semi_transp": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_couple_rayo_semi_transp(Coupled_problem):
    r"""
    Problem coupling several other problems to which radiation coupling is added (for semi
    transparent gas).

    You have to associate a modele_rayo_semi_transp

    You have to add a radiative term source in energy equation

    Warning: Calculation with semi transparent gas model may lead to divergence when high
    temperature differences are used. Indeed, the calculation of the stability time step of
    the equation does not take in account the source term. In semi transparent gas model,
    energy equation source term depends strongly of temperature via irradiance and stability
    is not guaranteed by the calculated time step. Reducing the facsec of the time scheme is a
    good tip to reach convergence when divergence is encountered.
    """

    _synonyms: ClassVar[dict] = {None: [], "groupes": []}


################################################################


class Source_rayo_semi_transp(Source_base):
    r"""
    Radiative term source in energy equation.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Flux_radiatif(Condlim_base):
    r"""
    Boundary condition for radiation equation.
    """

    na: Literal["a"] = Field(
        description=r"""Keyword for constant in boundary condition for irradiancy (sqrt(3) for half-infinite domain or 2 in closed domain).""",
        default="a",
    )
    a: float = Field(
        description=r"""Value of constant in boundary condition for irradiancy (sqrt(3) for half-infinite domain or 2 in closed domain).""",
        default=0.0,
    )
    ne: Literal["emissivite"] = Field(
        description=r"""Keyword for wall emissivity.""", default="emissivite"
    )
    emissivite: Front_field_base = Field(
        description=r"""Wall emissivity, value between 0 and 1.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "na": [],
        "a": [],
        "ne": [],
        "emissivite": [],
    }


################################################################


class Flux_radiatif_vdf(Flux_radiatif):
    r"""
    Boundary condition for radiation equation in VDF.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "na": [],
        "a": [],
        "ne": [],
        "emissivite": [],
    }


################################################################


class Flux_radiatif_vef(Flux_radiatif):
    r"""
    Boundary condition for radiation equation in VEF.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "na": [],
        "a": [],
        "ne": [],
        "emissivite": [],
    }


################################################################


class Paroi_contact_rayo(Condlim_base):
    r"""
    Thermal condition between two domains.
    """

    autrepb: str = Field(description=r"""Name of other problem.""", default="")
    nameb: str = Field(
        description=r"""boundary name of the remote problem which should be the same than the local name""",
        default="",
    )
    type: Literal["transp", "semi_transp"] = Field(
        description=r"""not_set""", default="transp"
    )
    _synonyms: ClassVar[dict] = {None: [], "autrepb": [], "nameb": [], "type": []}


################################################################


class Algo_base(Objet_u):
    r"""
    Basic class for multi-grid algorithms.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Algo_couple_1(Algo_base):
    r"""
    not_set
    """

    dt_uniforme: Optional[bool] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "dt_uniforme": []}


################################################################


class Pb_mg(Pb_gen_base):
    r"""
    Multi-grid problem.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Associate(Interprete):
    r"""
    This interpretor allows one object to be associated with another. The order of the two
    objects in this instruction is not important. The object objet_2 is associated to objet_1
    if this makes sense; if not either objet_1 is associated to objet_2 or the program exits
    with error because it cannot execute the Associate (Associer) instruction. For example, to
    calculate water flow in a pipe, a Pb_Hydraulique type object needs to be defined. But also
    a Domaine type object to represent the pipe, a Scheme_euler_explicit type object for time
    discretization, a discretization type object (VDF or VEF) and a Fluide_Incompressible type
    object which will contain the water properties. These objects must then all be associated
    with the problem.
    """

    objet_1: str = Field(description=r"""Objet_1""", default="")
    objet_2: str = Field(description=r"""Objet_2""", default="")
    _synonyms: ClassVar[dict] = {None: ["associer"], "objet_1": [], "objet_2": []}


################################################################


class Associer_pbmg_pbgglobal(Associate):
    r"""
    This interpretor allows a global problem to be associated with multi-grid problem.
    """

    _synonyms: ClassVar[dict] = {None: [], "objet_1": [], "objet_2": []}


################################################################


class Associer_pbmg_pbfin(Associate):
    r"""
    This interpretor allows a local problem to be associated with multi-grid problem.
    """

    _synonyms: ClassVar[dict] = {None: [], "objet_1": [], "objet_2": []}


################################################################


class Associer_algo(Associate):
    r"""
    This interpretor allows an algorithm to be associated with multi-grid problem.
    """

    _synonyms: ClassVar[dict] = {None: [], "objet_1": [], "objet_2": []}


################################################################


class Paroi_echange_contact_vdf_zoom_grossier(Paroi_echange_externe_impose):
    r"""
    External type exchange condition with a heat exchange coefficient and an imposed external
    temperature in the case of zoom (coarse).
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Paroi_echange_contact_vdf_zoom_fin(Paroi_echange_externe_impose):
    r"""
    External type exchange condition with a heat exchange coefficient and an imposed external
    temperature in the case of zoom (fine).
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Champ_front_zoom(Front_field_base):
    r"""
    Basic class for fields at boundaries of two problems (global problem and local problem).
    """

    pbmg: str = Field(description=r"""Name of multi-grid problem.""", default="")
    pb_1: str = Field(description=r"""Name of first problem.""", default="")
    pb_2: str = Field(description=r"""Name of second problem.""", default="")
    bord: str = Field(description=r"""Name of bord.""", default="")
    inco: str = Field(description=r"""Name of field.""", default="")
    _synonyms: ClassVar[dict] = {
        None: [],
        "pbmg": [],
        "pb_1": [],
        "pb_2": [],
        "bord": [],
        "inco": [],
    }


################################################################


class Contact_vdf_vef(Condlim_base):
    r"""
    Boundary condition in the case of two problems (VDF -> VEF).
    """

    champ: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "champ": []}


################################################################


class Contact_vef_vdf(Condlim_base):
    r"""
    Boundary condition in the case of two problems (VEF -> VDF).
    """

    champ: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "champ": []}


################################################################


class Methode_loi_horaire(Methode_transport_deriv):
    r"""
    not_set
    """

    nom_loi: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: ["loi_horaire"], "nom_loi": []}


################################################################


class Type_diffusion_turbulente_multiphase_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_diffusion_turbulente_multiphase_multiple_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_diffusion_turbulente_multiphase_multiple_k_omega(
    Type_diffusion_turbulente_multiphase_multiple_deriv
):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["k_omega"]}


################################################################


class Type_diffusion_turbulente_multiphase_multiple_sato(
    Type_diffusion_turbulente_multiphase_multiple_deriv
):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["sato"]}


################################################################


class Type_diffusion_turbulente_multiphase_multiple(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    See TrioCFD_Pb_multiphase.pdf
    """

    k_omega: Optional[Type_diffusion_turbulente_multiphase_multiple_k_omega] = Field(
        description=r"""first correlation""", default=None
    )
    sato: Optional[Type_diffusion_turbulente_multiphase_multiple_sato] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["multiple"], "k_omega": [], "sato": []}


################################################################


class Option_vdf(Interprete):
    r"""
    Class of VDF options.
    """

    traitement_coins: Optional[Literal["oui", "non"]] = Field(
        description=r"""Treatment of corners (yes or no). This option modifies slightly the calculations at the outlet of the plane channel. It supposes that the boundary continues after channel outlet (i.e. velocity vector remains parallel to the boundary).""",
        default=None,
    )
    traitement_gradients: Optional[Literal["oui", "non"]] = Field(
        description=r"""Treatment of gradient calculations (yes or no). This option modifies slightly the gradient calculation at the corners and activates also the corner treatment option.""",
        default=None,
    )
    p_imposee_aux_faces: Optional[Literal["oui", "non"]] = Field(
        description=r"""Pressure imposed at the faces (yes or no).""", default=None
    )
    all_options: Optional[bool] = Field(
        description=r"""Activates all Option_VDF options. If used, must be used alone without specifying the other options, nor combinations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "traitement_coins": [],
        "traitement_gradients": [],
        "p_imposee_aux_faces": [],
        "all_options": ["toutes_les_options"],
    }


################################################################


class Sous_maille_wale(Mod_turb_hyd_ss_maille):
    r"""
    This is the WALE-model. It is a new sub-grid scale model for eddy-viscosity in LES that
    has the following properties :

    - it goes naturally to 0 at the wall (it doesn\'t need any information on the wall
    position or geometry)

    - it has the proper wall scaling in o(y3) in the vicinity of the wall

    - it reproduces correctly the laminar to turbulent transition.
    """

    cw: Optional[float] = Field(
        description=r"""The unique parameter (constant) of the WALE-model (by default value 0.5).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "cw": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Sous_maille_smago(Mod_turb_hyd_ss_maille):
    r"""
    Smagorinsky sub-grid turbulence model.

    Nut=Cs1*Cs1*l*l*sqrt(2*S*S)

    K=Cs2*Cs2*l*l*2*S
    """

    cs: Optional[float] = Field(
        description=r"""This is an optional keyword and the value is used to set the constant used in the Smagorinsky model (This is currently only valid for Smagorinsky models and it is set to 0.18 by default) .""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "cs": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Dirac(Source_base):
    r"""
    Class to define a source term corresponding to a volume power release in the energy
    equation.
    """

    position: List[float] = Field(description=r"""not_set""", default_factory=list)
    ch: Field_base = Field(
        description=r"""Thermal power field type. To impose a volume power on a domain sub-area, the Champ_Uniforme_Morceaux (partly_uniform_field) type must be used.  Warning : The volume thermal power is expressed in W.m-3.""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "position": [], "ch": []}


################################################################


class Forchheimer(Source_base):
    r"""
    Class to add the source term of Forchheimer -Cf/sqrt(K)*V2 in the Navier-Stokes equations.
    We must precise a permeability model : constant or Ergun\'s law. Moreover we can give the
    constant Cf : by default its value is 1. Forchheimer source term is available also for
    quasi compressible calculation. A new keyword is aded for porosity (porosite).
    """

    bloc: Bloc_lecture = Field(
        description=r"""Description.""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Source_constituant(Source_base):
    r"""
    Keyword to specify source rates, in [[C]/s], for each one of the nb constituents. [C] is
    the concentration unit.
    """

    ch: Field_base = Field(
        description=r"""Field type.""", default_factory=lambda: eval("Field_base()")
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Echange_interne_parfait(Condlim_base):
    r"""
    Internal heat exchange boundary condition with perfect (infinite) exchange coefficient.
    """

    _synonyms: ClassVar[dict] = {None: ["paroi_echange_interne_parfait"]}


################################################################


class Paroi_echange_contact_correlation_vdf(Condlim_base):
    r"""
    Class to define a thermohydraulic 1D model which will apply to a boundary of 2D or 3D
    domain.

    Warning : For parallel calculation, the only possible partition will be according the
    axis of the model with the keyword Tranche.
    """

    dir: Optional[int] = Field(
        description=r"""Direction (0 : axis X, 1 : axis Y, 2 : axis Z) of the 1D model.""",
        default=None,
    )
    tinf: Optional[float] = Field(
        description=r"""Inlet fluid temperature of the 1D model (oC or K).""",
        default=None,
    )
    tsup: Optional[float] = Field(
        description=r"""Outlet fluid temperature of the 1D model (oC or K).""",
        default=None,
    )
    lambda_: Optional[str] = Field(
        description=r"""Thermal conductivity of the fluid (W.m-1.K-1).""", default=None
    )
    rho: Optional[str] = Field(
        description=r"""Mass density of the fluid (kg.m-3) which may be a function of the temperature T.""",
        default=None,
    )
    dt_impr: Optional[float] = Field(
        description=r"""Printing period in name_of_data_file_time.dat files of the 1D model results.""",
        default=None,
    )
    cp: Optional[float] = Field(
        description=r"""Calorific capacity value at a constant pressure of the fluid (J.kg-1.K-1).""",
        default=None,
    )
    mu: Optional[str] = Field(
        description=r"""Dynamic viscosity of the fluid (kg.m-1.s-1) which may be a function of thetemperature T.""",
        default=None,
    )
    debit: Optional[float] = Field(
        description=r"""Surface flow rate (kg.s-1.m-2) of the fluid into the channel.""",
        default=None,
    )
    dh: Optional[float] = Field(
        description=r"""Hydraulic diameter may be a function f(x) with x position along the 1D axis (xinf <= x <= xsup)""",
        default=None,
    )
    volume: Optional[str] = Field(
        description=r"""Exact volume of the 1D domain (m3) which may be a function of the hydraulic diameter (Dh) and the lateral surface (S) of the meshed boundary.""",
        default=None,
    )
    nu: Optional[str] = Field(
        description=r"""Nusselt number which may be a function of the Reynolds number (Re) and the Prandtl number (Pr).""",
        default=None,
    )
    reprise_correlation: Optional[bool] = Field(
        description=r"""Keyword in the case of a resuming calculation with this correlation.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dir": [],
        "tinf": [],
        "tsup": [],
        "lambda_": ["lambda"],
        "rho": [],
        "dt_impr": [],
        "cp": [],
        "mu": [],
        "debit": [],
        "dh": [],
        "volume": [],
        "nu": [],
        "reprise_correlation": [],
    }


################################################################


class Echange_interne_impose(Condlim_base):
    r"""
    Internal heat exchange boundary condition with exchange coefficient.
    """

    h_imp: str = Field(
        description=r"""Exchange coefficient value expressed in W.m-2.K-1.""",
        default="",
    )
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["paroi_echange_interne_impose"],
        "h_imp": [],
        "ch": [],
    }


################################################################


class Frontiere_ouverte_gradient_pression_impose(Neumann):
    r"""
    Normal imposed pressure gradient condition on the open boundary called bord (edge). This
    boundary condition may be only used in VDF discretization. The imposed $\partial
    P/\partial n$ value is expressed in Pa.m-1.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_pression_imposee_orlansky(Neumann):
    r"""
    This boundary condition may only be used with VDF discretization. There is no reference
    for pressure for this boundary condition so it is better to add pressure condition (with
    Frontiere_ouverte_pression_imposee) on one or two cells (for symetry in a channel) of the
    boundary where Orlansky conditions are imposed.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Discretisation_base(Objet_u):
    r"""
    Basic class for space discretization of thermohydraulic turbulent problems.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Vdf(Discretisation_base):
    r"""
    Finite difference volume discretization.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Champ_front_debit_qc_vdf_fonc_t(Front_field_base):
    r"""
    This keyword is used to define a flow rate field for quasi-compressible fluids in VDF
    discretization. The flow rate could be constant or time-dependent.
    """

    dimension: int = Field(description=r"""Problem dimension""", default=0)
    liste: Bloc_lecture = Field(
        description=r"""List of the mass flow rate values [kg/s/m2] with the following syntaxe: { val1 ... valdim } where val1 ... valdim are constant or function of time.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    moyen: Optional[str] = Field(
        description=r"""Option to use rho mean value""", default=None
    )
    pb_name: str = Field(description=r"""Problem name""", default="")
    _synonyms: ClassVar[dict] = {
        None: [],
        "dimension": ["dim"],
        "liste": [],
        "moyen": [],
        "pb_name": [],
    }


################################################################


class Champ_don_base(Field_base):
    r"""
    Basic class for data fields (not calculated), p.e. physics properties.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Champ_som_lu_vdf(Champ_don_base):
    r"""
    Keyword to read in a file values located at the nodes of a mesh in VDF discretization.
    """

    domain_name: str = Field(description=r"""Name of the domain.""", default="")
    dim: int = Field(description=r"""Value of the dimension of the field.""", default=0)
    tolerance: float = Field(
        description=r"""Value of the tolerance to check the coordinates of the nodes.""",
        default=0.0,
    )
    file: str = Field(
        description=r"""name of the file  This file has the following format:  Xi Yi Zi -> Coordinates of the node  Ui Vi Wi -> Value of the field on this node  Xi+1 Yi+1 Zi+1 -> Next point  Ui+1 Vi+1 Zi+1 -> Next value ...""",
        default="",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domain_name": [],
        "dim": [],
        "tolerance": [],
        "file": [],
    }


################################################################


class Champ_front_debit_qc_vdf(Front_field_base):
    r"""
    This keyword is used to define a flow rate field for quasi-compressible fluids in VDF
    discretization. The flow rate is kept constant during a transient.
    """

    dimension: int = Field(description=r"""Problem dimension""", default=0)
    liste: Bloc_lecture = Field(
        description=r"""List of the mass flow rate values [kg/s/m2] with the following syntaxe: { val1 ... valdim }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    moyen: Optional[str] = Field(
        description=r"""Option to use rho mean value""", default=None
    )
    pb_name: str = Field(description=r"""Problem name""", default="")
    _synonyms: ClassVar[dict] = {
        None: [],
        "dimension": ["dim"],
        "liste": [],
        "moyen": [],
        "pb_name": [],
    }


################################################################


class Profils_thermo(Traitement_particulier_base):
    r"""
    non documente
    """

    bloc: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Solveur_implicite_base(Objet_u):
    r"""
    Class for solver in the situation where the time scheme is the implicit scheme. Solver
    allows equation diffusion and convection operators to be set as implicit terms.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Solveur_lineaire_std(Solveur_implicite_base):
    r"""
    not_set
    """

    solveur: Optional[Solveur_sys_base] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": []}


################################################################


class Parametre_implicite(Parametre_equation_base):
    r"""
    Keyword to change for this equation only the parameter of the implicit scheme used to
    solve the problem.
    """

    seuil_convergence_implicite: Optional[float] = Field(
        description=r"""Keyword to change for this equation only the value of seuil_convergence_implicite used in the implicit scheme.""",
        default=None,
    )
    seuil_convergence_solveur: Optional[float] = Field(
        description=r"""Keyword to change for this equation only the value of seuil_convergence_solveur used in the implicit scheme""",
        default=None,
    )
    solveur: Optional[Solveur_sys_base] = Field(
        description=r"""Keyword to change for this equation only the solver used in the implicit scheme""",
        default=None,
    )
    resolution_explicite: Optional[bool] = Field(
        description=r"""To solve explicitly the equation whereas the scheme is an implicit scheme.""",
        default=None,
    )
    equation_non_resolue: Optional[bool] = Field(
        description=r"""Keyword to specify that the equation is not solved.""",
        default=None,
    )
    equation_frequence_resolue: Optional[str] = Field(
        description=r"""Keyword to specify that the equation is solved only every n time steps (n is an integer or given by a time-dependent function f(t)).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "seuil_convergence_solveur": [],
        "solveur": [],
        "resolution_explicite": [],
        "equation_non_resolue": [],
        "equation_frequence_resolue": [],
    }


################################################################


class Parametre_diffusion_implicite(Parametre_equation_base):
    r"""
    To specify additional parameters for the equation when using impliciting diffusion
    """

    crank: Optional[Literal[0, 1]] = Field(
        description=r"""Use (1) or not (0, default) a Crank Nicholson method for the diffusion implicitation algorithm. Setting crank to 1 increases the order of the algorithm from 1 to 2.""",
        default=None,
    )
    preconditionnement_diag: Optional[Literal[0, 1]] = Field(
        description=r"""The CG used to solve the implicitation of the equation diffusion operator is not preconditioned by default. If this option is set to 1, a diagonal preconditionning is used. Warning: this option is not necessarily more efficient, depending on the treated case.""",
        default=None,
    )
    niter_max_diffusion_implicite: Optional[int] = Field(
        description=r"""Change the maximum number of iterations for the CG (Conjugate Gradient) algorithm when solving the diffusion implicitation of the equation.""",
        default=None,
    )
    seuil_diffusion_implicite: Optional[float] = Field(
        description=r"""Change the threshold convergence value used by default for the CG resolution for the diffusion implicitation of this equation.""",
        default=None,
    )
    solveur: Optional[Solveur_sys_base] = Field(
        description=r"""Method (different from the default one, Conjugate Gradient) to solve the linear system.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "crank": [],
        "preconditionnement_diag": [],
        "niter_max_diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "solveur": [],
    }


################################################################


class Testeur_medcoupling(Interprete):
    r"""
    not_set
    """

    pb_name: str = Field(description=r"""Name of domain.""", default="")
    field_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "pb_name": [], "field_name": ["filed_name"]}


################################################################


class Pilote_icoco(Interprete):
    r"""
    not_set
    """

    pb_name: str = Field(description=r"""not_set""", default="")
    main: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: [], "pb_name": [], "main": []}


################################################################


class Postraiter_domaine(Interprete):
    r"""
    To write one or more domains in a file with a specified format
    (MED,LML,LATA,SINGLE_LATA,CGNS).
    """

    format: Literal["lml", "lata", "single_lata", "lata_v2", "med", "cgns"] = Field(
        description=r"""File format.""", default="lml"
    )
    binaire: Optional[Literal[0, 1]] = Field(
        description=r"""Binary (binaire 1) or ASCII (binaire 0) may be used. By default, it is 0 for LATA and only ASCII is available for LML and only binary is available for MED.""",
        default=None,
    )
    ecrire_frontiere: Optional[Literal[0, 1]] = Field(
        description=r"""This option will write (if set to 1, the default) or not (if set to 0) the boundaries as fields into the file (it is useful to not add the boundaries when writing a domain extracted from another domain)""",
        default=None,
    )
    fichier: Optional[str] = Field(
        description=r"""The file name can be changed with the fichier option.""",
        default=None,
    )
    joints_non_postraites: Optional[Literal[0, 1]] = Field(
        description=r"""The joints_non_postraites (1 by default) will not write the boundaries between the partitioned mesh.""",
        default=None,
    )
    domaine: Optional[str] = Field(description=r"""Name of domain""", default=None)
    domaines: Optional[Bloc_lecture] = Field(
        description=r"""Names of domains : { name1 name2 }""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "format": [],
        "binaire": [],
        "ecrire_frontiere": [],
        "fichier": ["file"],
        "joints_non_postraites": [],
        "domaine": ["domain"],
        "domaines": [],
    }


################################################################


class Link_cgns_files(Interprete):
    r"""
    Creates a single CGNS xxxx.cgns file that links to a xxxx.grid.cgns and
    xxxx.solution.*.cgns files
    """

    base_name: str = Field(
        description=r"""Base name of the gid/solution cgns files.""", default=""
    )
    output_name: str = Field(
        description=r"""Name of the output cgns file.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: [], "base_name": [], "output_name": []}


################################################################


class Merge_med(Interprete):
    r"""
    This keyword allows to merge multiple MED files produced during a parallel computation
    into a single MED file.
    """

    med_files_base_name: str = Field(
        description=r"""Base name of multiple med files that should appear as base_name_xxxxx.med, where xxxxx denotes the MPI rank number. If you specify NOM_DU_CAS, it will automatically take the basename from your datafile's name.""",
        default="",
    )
    time_iterations: Literal["all_times", "last_time"] = Field(
        description=r"""Identifies whether to merge all time iterations present in the MED files or only the last one.""",
        default="all_times",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "med_files_base_name": [],
        "time_iterations": [],
    }


################################################################


class Lml_to_lata(Interprete):
    r"""
    To convert results file written with LML format to a single LATA file.
    """

    file_lml: str = Field(
        description=r"""LML file to convert to the new format.""", default=""
    )
    file_lata: str = Field(description=r"""Name of the single LATA file.""", default="")
    _synonyms: ClassVar[dict] = {None: ["lml_2_lata"], "file_lml": [], "file_lata": []}


################################################################


class Format_lata_to_med(Objet_lecture):
    r"""
    not_set
    """

    mot: Literal["format_post_sup"] = Field(
        description=r"""not_set""", default="format_post_sup"
    )
    format: Optional[Literal["lml", "lata", "lata_v2", "med"]] = Field(
        description=r"""generated file post_med.data use format (MED or LATA or LML keyword).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": [], "format": []}


################################################################


class Lata_to_med(Interprete):
    r"""
    To convert results file written with LATA format to MED file. Warning: Fields located on
    faces are not supported yet.
    """

    format: Optional[Format_lata_to_med] = Field(
        description=r"""generated file post_med.data use format (MED or LATA or LML keyword).""",
        default=None,
    )
    file: str = Field(
        description=r"""LATA file to convert to the new format.""", default=""
    )
    file_med: str = Field(description=r"""Name of the MED file.""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["lata_2_med"],
        "format": [],
        "file": [],
        "file_med": [],
    }


################################################################


class Format_lata_to_cgns(Objet_lecture):
    r"""
    not_set
    """

    mot: Literal["format_post_sup"] = Field(
        description=r"""not_set""", default="format_post_sup"
    )
    format: Optional[Literal["lml", "lata", "lata_v2", "med", "cgns"]] = Field(
        description=r"""generated file post_CGNS.data use format (CGNS or LATA or LML keyword).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": [], "format": []}


################################################################


class Lata_to_other(Interprete):
    r"""
    To convert results file written with LATA format to CGNS, MED or LML format. Warning:
    Fields located at faces are not supported yet.
    """

    format: Optional[Literal["lml", "lata", "lata_v2", "med", "cgns"]] = Field(
        description=r"""Results format (CGNS, MED or LATA or LML keyword).""",
        default=None,
    )
    file: str = Field(
        description=r"""LATA file to convert to the new format.""", default=""
    )
    file_post: str = Field(description=r"""Name of file post.""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["lata_2_other"],
        "format": [],
        "file": [],
        "file_post": [],
    }


################################################################


class Option_cgns(Interprete):
    r"""
    Class for CGNS options.
    """

    single_precision: Optional[bool] = Field(
        description=r"""If used, data will be written with a single_precision format inside the CGNS file (it concerns both mesh coordinates and field values).""",
        default=None,
    )
    multiple_files: Optional[bool] = Field(
        description=r"""If used, data will be written in separate files (ie: one file per processor).""",
        default=None,
    )
    parallel_over_zone: Optional[bool] = Field(
        description=r"""If used, data will be written in separate zones (ie: one zone per processor). This is not so performant but easier to read later ...""",
        default=None,
    )
    use_links: Optional[bool] = Field(
        description=r"""If used, data will be written in separate files; one file for mesh, and then one file for solution time. Links will be used.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "single_precision": [],
        "multiple_files": [],
        "parallel_over_zone": [],
        "use_links": [],
    }


################################################################


class Stat_post_t_deb(Stat_post_deriv):
    r"""
    Start of integration time
    """

    val: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["t_deb"], "val": []}


################################################################


class Stat_post_t_fin(Stat_post_deriv):
    r"""
    End of integration time
    """

    val: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["t_fin"], "val": []}


################################################################


class Stat_post_moyenne(Stat_post_deriv):
    r"""
    to calculate the average of the field over time
    """

    field: str = Field(
        description=r"""name of the field on which statistical analysis will be performed. Possible keywords are Vitesse (velocity), Pression (pressure), Temperature, Concentration, ...""",
        default="",
    )
    localisation: Optional[Literal["elem", "som", "faces"]] = Field(
        description=r"""Localisation of post-processed field value""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["moyenne", "champ_post_statistiques_moyenne"],
        "field": [],
        "localisation": [],
    }


################################################################


class Stat_post_ecart_type(Stat_post_deriv):
    r"""
    to calculate the standard deviation (statistic rms) of the field
    """

    field: str = Field(
        description=r"""name of the field on which statistical analysis will be performed. Possible keywords are Vitesse (velocity), Pression (pressure), Temperature, Concentration, ...""",
        default="",
    )
    localisation: Optional[Literal["elem", "som", "faces"]] = Field(
        description=r"""Localisation of post-processed field value""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_statistiques_ecart_type", "ecart_type"],
        "field": [],
        "localisation": [],
    }


################################################################


class Stat_post_correlation(Stat_post_deriv):
    r"""
    correlation between the two fields
    """

    first_field: str = Field(description=r"""first field""", default="")
    second_field: str = Field(description=r"""second field""", default="")
    localisation: Optional[Literal["elem", "som", "faces"]] = Field(
        description=r"""Localisation of post-processed field value""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_statistiques_correlation", "correlation"],
        "first_field": [],
        "second_field": [],
        "localisation": [],
    }


################################################################


class Segmentfacesx(Sonde_base):
    r"""
    Segment probe where points are moved to the nearest x faces
    """

    nbr: int = Field(
        description=r"""Number of probe points of the segment, evenly distributed.""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""First outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin: Un_point = Field(
        description=r"""Second outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nbr": [], "point_deb": [], "point_fin": []}


################################################################


class Segmentfacesy(Sonde_base):
    r"""
    Segment probe where points are moved to the nearest y faces
    """

    nbr: int = Field(
        description=r"""Number of probe points of the segment, evenly distributed.""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""First outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin: Un_point = Field(
        description=r"""Second outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nbr": [], "point_deb": [], "point_fin": []}


################################################################


class Segmentfacesz(Sonde_base):
    r"""
    Segment probe where points are moved to the nearest z faces
    """

    nbr: int = Field(
        description=r"""Number of probe points of the segment, evenly distributed.""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""First outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin: Un_point = Field(
        description=r"""Second outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nbr": [], "point_deb": [], "point_fin": []}


################################################################


class Radius(Sonde_base):
    r"""
    not_set
    """

    nbr: int = Field(
        description=r"""Number of probe points of the segment, evenly distributed.""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""First outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    radius: float = Field(description=r"""not_set""", default=0.0)
    teta1: float = Field(description=r"""not_set""", default=0.0)
    teta2: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "nbr": [],
        "point_deb": [],
        "radius": [],
        "teta1": [],
        "teta2": [],
    }


################################################################


class Listpoints(Listobj):
    r"""
    Points.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Points(Sonde_base):
    r"""
    Keyword to define the number of probe points. The file is arranged in columns.
    """

    points: Annotated[List[Un_point], "Listpoints"] = Field(
        description=r"""Points.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "points": []}


################################################################


class Numero_elem_sur_maitre(Sonde_base):
    r"""
    Keyword to define a probe at the special element. Useful for min/max sonde.
    """

    numero: int = Field(description=r"""element number""", default=0)
    _synonyms: ClassVar[dict] = {None: [], "numero": []}


################################################################


class Segmentpoints(Points):
    r"""
    This keyword is used to define a probe segment from specifics points. The nom_champ field
    is sampled at ns specifics points.
    """

    _synonyms: ClassVar[dict] = {None: [], "points": []}


################################################################


class Position_like(Sonde_base):
    r"""
    Keyword to define a probe at the same position of another probe named autre_sonde.
    """

    autre_sonde: str = Field(description=r"""Name of the other probe.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "autre_sonde": []}


################################################################


class Plan(Sonde_base):
    r"""
    Keyword to set the number of probe layout points. The file format is type .lml
    """

    nbr: int = Field(
        description=r"""Number of probes in the first direction.""", default=0
    )
    nbr2: int = Field(
        description=r"""Number of probes in the second direction.""", default=0
    )
    point_deb: Un_point = Field(
        description=r"""First point defining the angle. This angle should be positive.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin: Un_point = Field(
        description=r"""Second point defining the angle. This angle should be positive.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin_2: Un_point = Field(
        description=r"""Third point defining the angle. This angle should be positive.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nbr": [],
        "nbr2": [],
        "point_deb": [],
        "point_fin": [],
        "point_fin_2": [],
    }


################################################################


class Volume(Sonde_base):
    r"""
    Keyword to define the probe volume in a parallelepiped passing through 4 points and the
    number of probes in each direction.
    """

    nbr: int = Field(
        description=r"""Number of probes in the first direction.""", default=0
    )
    nbr2: int = Field(
        description=r"""Number of probes in the second direction.""", default=0
    )
    nbr3: int = Field(
        description=r"""Number of probes in the third direction.""", default=0
    )
    point_deb: Un_point = Field(
        description=r"""Point of origin.""", default_factory=lambda: eval("Un_point()")
    )
    point_fin: Un_point = Field(
        description=r"""Point defining the first direction (from point of origin).""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin_2: Un_point = Field(
        description=r"""Point defining the second direction (from point of origin).""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin_3: Un_point = Field(
        description=r"""Point defining the third direction (from point of origin).""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nbr": [],
        "nbr2": [],
        "nbr3": [],
        "point_deb": [],
        "point_fin": [],
        "point_fin_2": [],
        "point_fin_3": [],
    }


################################################################


class Circle(Sonde_base):
    r"""
    Keyword to define several probes located on a circle.
    """

    nbr: int = Field(
        description=r"""Number of probes between teta1 and teta2 (angles given in degrees).""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""Center of the circle.""",
        default_factory=lambda: eval("Un_point()"),
    )
    direction: Optional[Literal[0, 1, 2]] = Field(
        description=r"""Axis normal to the circle plane (0:x axis, 1:y axis, 2:z axis).""",
        default=None,
    )
    radius: float = Field(description=r"""Radius of the circle.""", default=0.0)
    theta1: float = Field(description=r"""First angle.""", default=0.0)
    theta2: float = Field(description=r"""Second angle.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "nbr": [],
        "point_deb": [],
        "direction": [],
        "radius": [],
        "theta1": [],
        "theta2": [],
    }


################################################################


class Circle_3(Sonde_base):
    r"""
    Keyword to define several probes located on a circle (in 3-D space).
    """

    nbr: int = Field(
        description=r"""Number of probes between teta1 and teta2 (angles given in degrees).""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""Center of the circle.""",
        default_factory=lambda: eval("Un_point()"),
    )
    direction: Literal[0, 1, 2] = Field(
        description=r"""Axis normal to the circle plane (0:x axis, 1:y axis, 2:z axis).""",
        default=0,
    )
    radius: float = Field(description=r"""Radius of the circle.""", default=0.0)
    theta1: float = Field(description=r"""First angle.""", default=0.0)
    theta2: float = Field(description=r"""Second angle.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "nbr": [],
        "point_deb": [],
        "direction": [],
        "radius": [],
        "theta1": [],
        "theta2": [],
    }


################################################################


class Lata_to_cgns(Interprete):
    r"""
    To convert results file written with LATA format to CGNS file. Warning: Fields located on
    faces are not supported yet.
    """

    format: Optional[Format_lata_to_cgns] = Field(
        description=r"""generated file post_CGNS.data use format (CGNS or LATA or LML keyword).""",
        default=None,
    )
    file: str = Field(
        description=r"""LATA file to convert to the new format.""", default=""
    )
    file_cgns: str = Field(description=r"""Name of the CGNS file.""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["lata_2_cgns"],
        "format": [],
        "file": [],
        "file_cgns": [],
    }


################################################################


class Ecrire_med_32_64(Interprete):
    r"""
    Write a domain to MED format into a file.
    """

    nom_dom: str = Field(description=r"""Name of domain.""", default="")
    file: str = Field(description=r"""Name of file.""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["write_med", "ecrire_med"],
        "nom_dom": [],
        "file": [],
    }


################################################################


class Ecrire_champ_med(Interprete):
    r"""
    Keyword to write a field to MED format into a file.
    """

    nom_dom: str = Field(description=r"""domain name""", default="")
    nom_chp: str = Field(description=r"""field name""", default="")
    file: str = Field(description=r"""file name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "nom_chp": [], "file": []}


################################################################


class Ecriturelecturespecial(Interprete):
    r"""
    Class to write or not to write a .xyz file on the disk at the end of the calculation.
    """

    type: str = Field(
        description=r"""If set to 0, no xyz file is created. If set to EFichierBin, it uses prior 1.7.0 way of reading xyz files (now LecFicDiffuseBin). If set to EcrFicPartageBin, it uses prior 1.7.0 way of writing xyz files (now EcrFicPartageMPIIO).""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "type": []}


################################################################


class Fonction_champ_reprise(Objet_lecture):
    r"""
    not_set
    """

    mot: Literal["fonction"] = Field(description=r"""not_set""", default="fonction")
    fonction: List[str] = Field(
        description=r"""n f1(val) f2(val) ... fn(val)] time""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": [], "fonction": []}


################################################################


class Champ_fonc_reprise(Champ_don_base):
    r"""
    This field is used to read a data field in a save file (.xyz or .sauv) at a specified
    time. It is very useful, for example, to run a thermohydraulic calculation with velocity
    initial condition read into a save file from a previous hydraulic calculation.
    """

    format: Optional[Literal["binaire", "formatte", "xyz", "single_hdf"]] = Field(
        description=r"""Type of file (the file format). If xyz format is activated, the .xyz file from the previous calculation will be given for filename, and if formatte or binaire is choosen, the .sauv file of the previous calculation will be specified for filename. In the case of a parallel calculation, if the mesh partition does not changed between the previous calculation and the next one, the binaire format should be preferred, because is faster than the xyz format. If single_hdf is used, the same constraints/advantages as binaire apply, but a single (HDF5) file is produced on the filesystem instead of having one file per processor.""",
        default=None,
    )
    filename: str = Field(description=r"""Name of the save file.""", default="")
    pb_name: str = Field(description=r"""Name of the problem.""", default="")
    champ: str = Field(
        description=r"""Name of the problem unknown. It may also be the temporal average of a problem unknown (like moyenne_vitesse, moyenne_temperature,...)""",
        default="",
    )
    fonction: Optional[Fonction_champ_reprise] = Field(
        description=r"""Optional keyword to apply a function on the field being read in the save file (e.g. to read a temperature field in Celsius units and convert it for the calculation on Kelvin units, you will use: fonction 1 273.+val )""",
        default=None,
    )
    temps: str = Field(
        description=r"""Time of the saved field in the save file or last_time. If you give the keyword last_time instead, the last time saved in the save file will be used.""",
        default="",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "format": [],
        "filename": [],
        "pb_name": [],
        "champ": [],
        "fonction": [],
        "temps": ["time"],
    }


################################################################


class Champ_front_fonc_xyz(Front_field_base):
    r"""
    Boundary field which is not constant in space.
    """

    val: List[str] = Field(
        description=r"""Values of field components (mathematical expressions).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Pb_champ_evaluateur(Objet_u):
    r"""
    specifies problem name, the field name beloging to the problem and number of field
    components.
    """

    pb: str = Field(
        description=r"""name of the problem where the source fields will be searched.""",
        default="",
    )
    champ: str = Field(description=r"""name of the field""", default="")
    ncomp: int = Field(description=r"""number of components""", default=0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "pb": [], "champ": [], "ncomp": []}


################################################################


class Moyenne_imposee_deriv(Objet_u):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Champ_front_recyclage(Front_field_base):
    r"""
    This keyword is used on a boundary to get a field from another boundary.

    It is to use, in a general way, on a boundary of a local_pb problem, a field calculated
    from a linear combination of an imposed field g(x,y,z,t) with an instantaneous f(x,y,z,t)
    and a spatial mean field <f>(t) or a temporal mean field <f>(x,y,z) extracted from a plane
    of a problem named pb (pb may be local_pb itself):

    For each component i, the field F applied on the boundary will be:

    F_i(x,y,z,t) = alpha_i*g_i(x,y,z,t) + xsi_i*[f_i(x,y,z,t)- beta_i*<fi>]
    """

    pb_champ_evaluateur: Pb_champ_evaluateur = Field(
        description=r"""not_set""",
        default_factory=lambda: eval("Pb_champ_evaluateur()"),
    )
    distance_plan: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""Vector which gives the distance between the boundary and the plane from where the field F will be extracted. By default, the vector is zero, that should imply the two domains have coincident boundaries.""",
        default=None,
    )
    ampli_moyenne_imposee: Optional[List[float]] = Field(
        description=r"""2|3 alpha(0) alpha(1) [alpha(2)]: alpha_i coefficients (by default =1)""",
        default=None,
    )
    ampli_moyenne_recyclee: Optional[List[float]] = Field(
        description=r"""2|3 beta(0) beta(1) [beta(2)]}: beta_i coefficients (by default =1)""",
        default=None,
    )
    ampli_fluctuation: Optional[List[float]] = Field(
        description=r"""2|3 gamma(0) gamma(1) [gamma(2)]}: gamma_i coefficients (by default =1)""",
        default=None,
    )
    direction_anisotrope: Optional[Literal[1, 2, 3]] = Field(
        description=r"""If an integer is given for direction (X:1, Y:2, Z:3, by default, direction is negative), the imposed field g will be 0 for the 2 other directions.""",
        default=None,
    )
    moyenne_imposee: Optional[Moyenne_imposee_deriv] = Field(
        description=r"""Value of the imposed g field.""", default=None
    )
    moyenne_recyclee: Optional[str] = Field(
        description=r"""Method used to perform a spatial or a temporal averaging of f field to specify <f>. <f> can be the surface mean of f on the plane (surface option, see below) or it can be read from several files (for example generated by the chmoy_faceperio option of the Traitement_particulier keyword to obtain a temporal mean field). The option methode_recyc can be: surfacique, Surface mean for <f> from f values on the plane ; Or one of the following methode_moy options applied to read a temporal mean field <f>(x,y,z): interpolation, connexion_approchee or connexion_exacte""",
        default=None,
    )
    fichier: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "pb_champ_evaluateur": [],
        "distance_plan": [],
        "ampli_moyenne_imposee": [],
        "ampli_moyenne_recyclee": [],
        "ampli_fluctuation": [],
        "direction_anisotrope": [],
        "moyenne_imposee": [],
        "moyenne_recyclee": [],
        "fichier": [],
    }


################################################################


class Moyenne_imposee_profil(Moyenne_imposee_deriv):
    r"""
    To specify analytic profile for the imposed g field.
    """

    profile: List[str] = Field(
        description=r"""specifies the analytic profile: 2|3 valx(x,y,z,t) valy(x,y,z,t) [valz(x,y,z,t)]""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: ["profil"], "profile": []}


################################################################


class Moyenne_imposee_connexion_exacte(Moyenne_imposee_deriv):
    r"""
    To read the imposed field from two files.
    """

    fichier: Literal["fichier"] = Field(description=r"""not_set""", default="fichier")
    file1: str = Field(
        description=r"""first file, contains the points coordinates (which should be the same as the coordinates of the boundary faces). The format of this file is: N  1 x(1) y(1) [z(1)]  2 x(2) y(2) [z(2)]  ...  N x(N) y(N) [z(N)]""",
        default="",
    )
    file2: Optional[str] = Field(
        description=r"""second file, contains the mean values. The format of this file is:  N  1 valx(1) valy(1) [valz(1)]  2 valx(2) valy(2) [valz(2)]  ...  N valx(N) valy(N) [valz(N)]""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["connexion_exacte"],
        "fichier": [],
        "file1": [],
        "file2": [],
    }


################################################################


class Moyenne_imposee_connexion_approchee(Moyenne_imposee_deriv):
    r"""
    To read the imposed field from a file where positions and values are given (it is not
    necessary that the coordinates of points match the coordinates of the boundary faces,
    indeed, the nearest point of each face of the boundary will be used).
    """

    fichier: Literal["fichier"] = Field(description=r"""not_set""", default="fichier")
    file1: str = Field(
        description=r"""filename. The format of the file is:  N  x(1) y(1) [z(1)] valx(1) valy(1) [valz(1)]  x(2) y(2) [z(2)] valx(2) valy(2) [valz(2)]  ...  x(N) y(N) [z(N)] valx(N) valy(N) [valz(N)]""",
        default="",
    )
    _synonyms: ClassVar[dict] = {
        None: ["connexion_approchee"],
        "fichier": [],
        "file1": [],
    }


################################################################


class Moyenne_imposee_interpolation(Moyenne_imposee_deriv):
    r"""
    To create an imposed field built by interpolation of values read from a file. The imposed
    field is applied on the direction given by the keyword direction_anisotrope (the field is
    zero for the other directions).
    """

    fichier: Literal["fichier"] = Field(
        description=r"""The format of the file is:  pos(1) val(1)  pos(2) val(2)  ...  pos(N) val(N)  If direction given by direction\_anisotrope is 1 (or 2 or 3), then pos will be X (or Y or Z) coordinate and val will be X value (or Y value, or Z value) of the imposed field.""",
        default="fichier",
    )
    file1: str = Field(description=r"""name of geom_face_perio""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_interpolation", "interpolation"],
        "fichier": [],
        "file1": [],
    }


################################################################


class Moyenne_imposee_logarithmique(Moyenne_imposee_deriv):
    r"""
    To specify the imposed field (in this case, velocity) by an analytical logarithmic law of
    the wall:

    g(x,y,z) = u_tau * ( log(0.5*diametre*u_tau/visco_cin)/Kappa + 5.1 )

    with g(x,y,z)=u(x,y,z) if direction is set to 1, g=v(x,y,z) if direction is set to 2 and
    g=w(w,y,z) if it is set to 3
    """

    diametre: Literal["diametre"] = Field(
        description=r"""not_set""", default="diametre"
    )
    val: float = Field(description=r"""diameter""", default=0.0)
    u_tau: Literal["u_tau"] = Field(description=r"""not_set""", default="u_tau")
    val_u_tau: float = Field(description=r"""value of u_tau""", default=0.0)
    visco_cin: Literal["visco_cin"] = Field(
        description=r"""not_set""", default="visco_cin"
    )
    val_visco_cin: float = Field(description=r"""value of visco_cin""", default=0.0)
    direction: Literal["direction"] = Field(
        description=r"""not_set""", default="direction"
    )
    val_direction: int = Field(description=r"""direction""", default=0)
    _synonyms: ClassVar[dict] = {
        None: ["logarithmique"],
        "diametre": [],
        "val": [],
        "u_tau": [],
        "val_u_tau": ["val_u_taul"],
        "visco_cin": [],
        "val_visco_cin": [],
        "direction": [],
        "val_direction": [],
    }


################################################################


class Disable_tu(Interprete):
    r"""
    Flag to disable the writing of the .TU files
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Nom(Objet_u):
    r"""
    Class to name the TRUST objects.
    """

    mot: Optional[str] = Field(description=r"""Chain of characters.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "mot": []}


################################################################


class Write_file(Interprete):
    r"""
    Keyword to write the object of name name_obj to a file filename. Since the v1.6.3, the
    default format is now binary format file.
    """

    name_obj: str = Field(
        description=r"""Name of the object to be written.""", default=""
    )
    filename: str = Field(description=r"""Name of the file.""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["ecrire_fichier", "ecrire_fichier_bin"],
        "name_obj": [],
        "filename": [],
    }


################################################################


class Execute_parallel(Interprete):
    r"""
    This keyword allows to run several computations in parallel on processors allocated to
    TRUST. The set of processors is split in N subsets and each subset will read and execute a
    different data file. Error messages usualy written to stderr and stdout are redirected to
    .log files (journaling must be activated).
    """

    liste_cas: List[str] = Field(
        description=r"""N datafile1 ... datafileN. datafileX the name of a TRUST data file without the .data extension.""",
        default_factory=list,
    )
    nb_procs: Optional[List[int]] = Field(
        description=r"""nb_procs is the number of processors needed to run each data file. If not given, TRUST assumes that computations are sequential.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "liste_cas": [], "nb_procs": []}


################################################################


class Option_interpolation(Interprete):
    r"""
    Class for interpolation fields using MEDCoupling.
    """

    sans_dec: Optional[bool] = Field(
        description=r"""Use remapper even for a parallel calculation""", default=None
    )
    sharing_algo: Optional[int] = Field(
        description=r"""Setting the DEC sharing algo : 0,1,2""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "sans_dec": ["without_dec"],
        "sharing_algo": [],
    }


################################################################


class Nom_anonyme(Nom):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["nul"], "mot": []}


################################################################


class Vect_nom(Listobj):
    r"""
    Vect of name.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class List_nom(Listobj):
    r"""
    List of name.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class List_nom_virgule(Listobj):
    r"""
    List of name.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class List_un_pb(Listobj):
    r"""
    pour les groupes
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Read_file(Interprete):
    r"""
    Keyword to read the object name_obj contained in the file filename.

    This is notably used when the calculation domain has already been meshed and the mesh
    contains the file filename, simply write read_file dom filename (where dom is the name of
    the meshed domain).

    If the filename is ;, is to execute a data set given in the file of name name_obj (a
    space must be entered between the semi-colon and the file name).
    """

    name_obj: str = Field(description=r"""Name of the object to be read.""", default="")
    filename: str = Field(description=r"""Name of the file.""", default="")
    _synonyms: ClassVar[dict] = {None: ["lire_fichier"], "name_obj": [], "filename": []}


################################################################


class Read_unsupported_ascii_file_from_icem(Read_file):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: [], "name_obj": [], "filename": []}


################################################################


class Ecrire_fichier_formatte(Write_file):
    r"""
    Keyword to write the object of name name_obj to a file filename in ASCII format.
    """

    _synonyms: ClassVar[dict] = {None: [], "name_obj": [], "filename": []}


################################################################


class Stat_per_proc_perf_log(Interprete):
    r"""
    Keyword allowing to activate the detailed statistics per processor (by default this is
    false, and only the master proc will produce stats).
    """

    flg: int = Field(
        description=r"""A flag that can be either 0 or 1 to turn off (default) or on the detailed stats.""",
        default=0,
    )
    _synonyms: ClassVar[dict] = {None: [], "flg": []}


################################################################


class Read_file_bin(Read_file):
    r"""
    Keyword to read an object name_obj in the unformatted type file filename.
    """

    _synonyms: ClassVar[dict] = {
        None: ["read_file_binary", "lire_fichier_bin"],
        "name_obj": [],
        "filename": [],
    }


################################################################


class System(Interprete):
    r"""
    To run Unix commands from the data file. Example: System \'echo The End | mail
    trust@cea.fr\'
    """

    cmd: str = Field(description=r"""command to execute.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "cmd": []}


################################################################


class Read(Interprete):
    r"""
    Interpretor to read the a_object objet defined between the braces.
    """

    a_object: str = Field(description=r"""Object to be read.""", default="")
    bloc: str = Field(description=r"""Definition of the object.""", default="")
    _synonyms: ClassVar[dict] = {None: ["lire"], "a_object": [], "bloc": []}


################################################################


class Write(Interprete):
    r"""
    Keyword to write the object of name name_obj to a standard outlet.
    """

    name_obj: str = Field(
        description=r"""Name of the object to be written.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: ["ecrire"], "name_obj": []}


################################################################


class Multiplefiles(Interprete):
    r"""
    Change MPI rank limit for multiple files during I/O
    """

    type: int = Field(description=r"""New MPI rank limit""", default=0)
    _synonyms: ClassVar[dict] = {None: [], "type": []}


################################################################


class Fin(Interprete):
    r"""
    Keyword which must complete the data file. The execution of the data file stops when
    reaching this keyword.
    """

    _synonyms: ClassVar[dict] = {None: ["end"]}


################################################################


class Paroi_echange_global_impose(Condlim_base):
    r"""
    Global type exchange condition (internal) that is to say that diffusion on the first fluid
    mesh is not taken into consideration.
    """

    h_imp: str = Field(
        description=r"""Global exchange coefficient value. The global exchange coefficient value is expressed in W.m-2.K-1.""",
        default="",
    )
    himpc: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    text: str = Field(
        description=r"""External temperature value. The external temperature value is expressed in oC or K.""",
        default="",
    )
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "h_imp": [],
        "himpc": [],
        "text": [],
        "ch": [],
    }


################################################################


class Paroi_adiabatique(Condlim_base):
    r"""
    Normal zero flux condition at the wall called bord (edge).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Neumann_homogene(Condlim_base):
    r"""
    Homogeneous neumann boundary condition
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_contact_fictif(Condlim_base):
    r"""
    This keyword is derivated from paroi_contact and is especially dedicated to compute
    coupled fluid/solid/fluid problem in case of thin material. Thanks to this option, solid
    is considered as a fictitious media (no mesh, no domain associated), and coupling is
    performed by considering instantaneous thermal equilibrium in it (for the moment).
    """

    autrepb: str = Field(description=r"""Name of other problem.""", default="")
    nameb: str = Field(description=r"""Name of bord.""", default="")
    conduct_fictif: float = Field(description=r"""thermal conductivity""", default=0.0)
    ep_fictive: float = Field(
        description=r"""thickness of the fictitious media""", default=0.0
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "autrepb": [],
        "nameb": [],
        "conduct_fictif": [],
        "ep_fictive": [],
    }


################################################################


class Scalaire_impose_paroi(Dirichlet):
    r"""
    Imposed temperature condition at the wall called bord (edge).
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Symetrie(Condlim_base):
    r"""
    1). For Navier-Stokes equations, this keyword is used to designate a symmetry condition
    concerning the velocity at the boundary called bord (edge) (normal velocity at the edge
    equal to zero and tangential velocity gradient at the edge equal to zero); 2). For scalar
    transport equation, this keyword is used to set a symmetry condition on scalar on the
    boundary named bord (edge).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Echange_externe_radiatif(Condlim_base):
    r"""
    Combines radiative $(sigma * eps * (T^4 - T_ext^4))$ and convective $(h * (T - T_ext))$
    heat transfer boundary conditions, where sigma is the Stefan-Boltzmann constant, eps is
    the emi
    """

    h_imp: Literal["h_imp", "t_ext", "emissivite"] = Field(
        description=r"""Heat exchange coefficient value (expressed in W.m-2.K-1).""",
        default="h_imp",
    )
    himpc: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    emissivite: Literal["emissivite", "h_imp", "t_ext"] = Field(
        description=r"""Emissivity coefficient value.""", default="emissivite"
    )
    emissivitebc: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    t_ext: Literal["t_ext", "h_imp", "emissivite"] = Field(
        description=r"""External temperature value (expressed in oC or K).""",
        default="t_ext",
    )
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    temp_unit: Literal["temperature_unit"] = Field(
        description=r"""Temperature unit""", default="temperature_unit"
    )
    temp_unit_val: Literal["kelvin", "celsius"] = Field(
        description=r"""Temperature unit""", default="kelvin"
    )
    _synonyms: ClassVar[dict] = {
        None: ["paroi_echange_externe_radiatif"],
        "h_imp": [],
        "himpc": [],
        "emissivite": [],
        "emissivitebc": [],
        "t_ext": [],
        "ch": [],
        "temp_unit": [],
        "temp_unit_val": [],
    }


################################################################


class Paroi_contact(Condlim_base):
    r"""
    Thermal condition between two domains. Important: the name of the boundaries in the two
    domains should be the same. (Warning: there is also an old limitation not yet fixed on the
    sequential algorithm in VDF to detect the matching faces on the two boundaries: faces
    should be ordered in the same way). The kind of condition depends on the discretization.
    In VDF, it is a heat exchange condition, and in VEF, a temperature condition.

    Such a coupling requires coincident meshes for the moment. In case of non-coincident
    meshes, run is stopped and two external files are automatically generated in VEF
    (connectivity_failed_boundary_name and connectivity_failed_pb_name.med). In 2D, the
    keyword Decouper_bord_coincident associated to the connectivity_failed_boundary_name file
    allows to generate a new coincident mesh.

    In 3D, for a first preliminary cut domain with HOMARD (fluid for instance), the second
    problem associated to pb_name (solide in a fluid/solid coupling problem) has to be
    submitted to HOMARD cutting procedure with connectivity_failed_pb_name.med.

    Such a procedure works as while the primary refined mesh (fluid in our example) impacts
    the fluid/solid interface with a compact shape as described below (values 2 or 4 indicates
    the number of division from primary faces obtained in fluid domain at the interface after
    HOMARD cutting):

    2-2-2-2-2-2

    2-4-4-4-4-4-2 \\; 2-2-2

    2-4-4-4-4-2 \\; 2-4-2

    2-2-2-2-2 \\; 2-2

    OK


    2-2 \\; \\; 2-2-2

    2-4-2 \\; 2-2

    2-2 \\; 2-2

    NOT OK
    """

    autrepb: str = Field(description=r"""Name of other problem.""", default="")
    nameb: str = Field(
        description=r"""boundary name of the remote problem which should be the same than the local name""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "autrepb": [], "nameb": []}


################################################################


class Neumann_paroi(Condlim_base):
    r"""
    Neumann boundary condition for mass equation (multiphase problem)
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_defilante(Dirichlet):
    r"""
    Keyword to designate a condition where tangential velocity is imposed on the wall called
    bord (edge). If the velocity components set by the user is not tangential, projection is
    used.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_vitesse_imposee(Dirichlet):
    r"""
    Class for velocity-inlet boundary condition. The imposed velocity field at the inlet is
    vectorial and the imposed velocity values are expressed in m.s-1.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_vitesse_imposee_sortie(Frontiere_ouverte_vitesse_imposee):
    r"""
    Sub-class for velocity boundary condition. The imposed velocity field at the open boundary
    is vectorial and the imposed velocity values are expressed in m.s-1.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_alpha_impose(Dirichlet):
    r"""
    Imposed alpha condition at the open boundary.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_fraction_massique_imposee(Condlim_base):
    r"""
    not_set
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Periodic(Condlim_base):
    r"""
    1). For Navier-Stokes equations, this keyword is used to indicate that the horizontal
    inlet velocity values are the same as the outlet velocity values, at every moment. As
    regards meshing, the inlet and outlet edges bear the same name.; 2). For scalar transport
    equation, this keyword is used to set a periodic condition on scalar. The two edges
    dealing with this periodic condition bear the same name.
    """

    _synonyms: ClassVar[dict] = {None: ["periodique"]}


################################################################


class Decoupebord_pour_rayonnement(Interprete):
    r"""
    To subdivide the external boundary of a domain into several parts (may be useful for
    better accuracy when using radiation model in transparent medium). To specify the
    boundaries of the fine_domain_name domain to be splitted. These boundaries will be cut
    according the coarse mesh defined by either the keyword domaine_grossier (each boundary
    face of the coarse mesh coarse_domain_name will be used to group boundary faces of the
    fine mesh to define a new boundary), either by the keyword nb_parts_naif (each boundary of
    the fine mesh is splitted into a partition with nx*ny*nz elements), either by a geometric
    condition given by a formulae with the keyword condition_geometrique. If used, the
    coarse_domain_name domain should have the same boundaries name of the fine_domain_name
    domain.

    A mesh file (ASCII format, except if binaire option is specified) named by default
    newgeom (or specified by the nom_fichier_sortie keyword) will be created and will contain
    the fine_domain_name domain with the splitted boundaries named boundary_name%I (where I is
    between from 0 and n-1). Furthermore, several files named boundary_name%I and
    boundary_name_xv will be created, containing the definition of the subdived boundaries.
    newgeom will be used to calculate view factors with geom2ansys script whereas only the
    boundary_name_xv files will be necessary for the radiation calculation. The file listb
    will contain the list of the boundaries boundary_name%I.
    """

    domaine: str = Field(description=r"""not_set""", default="")
    domaine_grossier: Optional[str] = Field(description=r"""not_set""", default=None)
    nb_parts_naif: Optional[List[int]] = Field(description=r"""not_set""", default=None)
    nb_parts_geom: Optional[List[int]] = Field(description=r"""not_set""", default=None)
    condition_geometrique: Optional[List[str]] = Field(
        description=r"""not_set""", default=None
    )
    bords_a_decouper: List[str] = Field(
        description=r"""not_set""", default_factory=list
    )
    nom_fichier_sortie: Optional[str] = Field(description=r"""not_set""", default=None)
    binaire: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["decoupebord"],
        "domaine": [],
        "domaine_grossier": [],
        "nb_parts_naif": [],
        "nb_parts_geom": [],
        "condition_geometrique": [],
        "bords_a_decouper": [],
        "nom_fichier_sortie": [],
        "binaire": [],
    }


################################################################


class Extruder(Interprete):
    r"""
    Class to create a 3D tetrahedral/hexahedral mesh (a prism is cut in 14) from a 2D
    triangular/quadrangular mesh.
    """

    domaine: str = Field(description=r"""Name of the domain.""", default="")
    nb_tranches: int = Field(
        description=r"""Number of elements in the extrusion direction.""", default=0
    )
    direction: Troisf = Field(
        description=r"""Direction of the extrude operation.""",
        default_factory=lambda: eval("Troisf()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": ["domain_name"],
        "nb_tranches": [],
        "direction": [],
    }


################################################################


class Extruder_en3(Extruder):
    r"""
    Class to create a 3D tetrahedral/hexahedral mesh (a prism is cut in 3) from a 2D
    triangular/quadrangular mesh. The names of the boundaries (by default, devant (front) and
    derriere (back)) may be edited by the keyword nom_cl_devant and nom_cl_derriere. If 'null'
    is written for nom_cl, then no boundary condition is generated at this place.

    Recommendation : to ensure conformity between meshes (in case of fluid/solid coupling) it
    is recommended to extrude all the domains at the same time.
    """

    domaine: List[str] = Field(
        description=r"""List of the domains""", default_factory=list
    )
    nom_cl_devant: Optional[str] = Field(
        description=r"""New name of the first boundary.""", default=None
    )
    nom_cl_derriere: Optional[str] = Field(
        description=r"""New name of the second boundary.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": ["domain_name"],
        "nom_cl_devant": [],
        "nom_cl_derriere": [],
        "nb_tranches": [],
        "direction": [],
    }


################################################################


class Triangulate(Interprete):
    r"""
    To achieve a triangular mesh from a mesh comprising rectangles (2 triangles per
    rectangle). Should be used in VEF discretization. Principle:

    \includepng{{trianguler.pdf}}{{10}}
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: ["trianguler"], "domain_name": []}


################################################################


class Trianguler_fin(Triangulate):
    r"""
    Trianguler_fin is the recommended option to triangulate rectangles.

    As an extension (subdivision) of Triangulate_h option, this one cut each initial
    rectangle in 8 triangles (against 4, previously). This cutting ensures :

    - a correct cutting in the corners (in respect to pressure discretization PreP1B).

    - a better isotropy of elements than with Trianguler_h option.

    - a better alignment of summits (this could have a benefit effect on calculation near
    walls since first elements in contact with it are all contained in the same constant
    thickness, and, by this way, a 2D cartesian grid based on summits can be engendered and
    used to realize statistical analysis in plane channel configuration for instance).
    Principle:

    \includepng{{triangulerfin.pdf}}{{10}}
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Extract_2d_from_3d(Interprete):
    r"""
    Keyword to extract a 2D mesh by selecting a boundary of the 3D mesh. To generate a 2D
    axisymmetric mesh prefer Extract_2Daxi_from_3D keyword.
    """

    dom3d: str = Field(description=r"""Domain name of the 3D mesh""", default="")
    bord: str = Field(
        description=r"""Boundary name. This boundary becomes the new 2D mesh and all the boundaries, in 3D, attached to the selected boundary, give their name to the new boundaries, in 2D.""",
        default="",
    )
    dom2d: str = Field(description=r"""Domain name of the new 2D mesh""", default="")
    _synonyms: ClassVar[dict] = {None: [], "dom3d": [], "bord": [], "dom2d": []}


################################################################


class Extract_2daxi_from_3d(Extract_2d_from_3d):
    r"""
    Keyword to extract a 2D axisymetric mesh by selecting a boundary of the 3D mesh.
    """

    _synonyms: ClassVar[dict] = {None: [], "dom3d": [], "bord": [], "dom2d": []}


################################################################


class Imprimer_flux(Interprete):
    r"""
    This keyword prints the flux per face at the specified domain boundaries in the data set.
    The fluxes are written to the .face files at a frequency defined by dt_impr, the
    evaluation printing frequency (refer to time scheme keywords). By default, fluxes are
    incorporated onto the edges before being displayed.
    """

    domain_name: str = Field(description=r"""Name of the domain.""", default="")
    noms_bord: Bloc_lecture = Field(
        description=r"""List of boundaries, for ex: { Bord1 Bord2 }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "noms_bord": []}


################################################################


class Imprimer_flux_sum(Imprimer_flux):
    r"""
    This keyword prints the sum of the flux per face at the domain boundaries defined by the
    user in the data set. The fluxes are written into the .out files at a frequency defined by
    dt_impr, the evaluation printing frequency (refer to time scheme keywords).
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "noms_bord": []}


################################################################


class Remove_elem_bloc(Objet_lecture):
    r"""
    not_set
    """

    liste: Optional[List[int]] = Field(description=r"""not_set""", default=None)
    fonction: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["nul"], "liste": [], "fonction": []}


################################################################


class Remove_elem(Interprete):
    r"""
    Keyword to remove element from a VDF mesh (named domaine_name), either from an explicit
    list of elements or from a geometric condition defined by a condition f(x,y)>0 in 2D and
    f(x,y,z)>0 in 3D. All the new borders generated are gathered in one boundary called :
    newBord (to rename it, use RegroupeBord keyword. To split it to different boundaries, use
    DecoupeBord_Pour_Rayonnement keyword). Example of a removed zone of radius 0.2 centered at
    (x,y)=(0.5,0.5):

    Remove_elem dom { fonction $0.2*0.2-(x-0.5)^2-(y-0.5)^2>0$ }

    Warning : the thickness of removed zone has to be large enough to avoid singular nodes as
    decribed below : \includepng{{removeelem.jpeg}}{{11.234}}
    """

    domaine: str = Field(description=r"""Name of domain""", default="")
    bloc: Remove_elem_bloc = Field(
        description=r"""not_set""", default_factory=lambda: eval("Remove_elem_bloc()")
    )
    _synonyms: ClassVar[dict] = {None: [], "domaine": ["domain"], "bloc": []}


################################################################


class Raffiner_isotrope_parallele(Interprete):
    r"""
    Refine parallel mesh in parallel
    """

    name_of_initial_domaines: str = Field(
        description=r"""name of initial Domaines""", default=""
    )
    name_of_new_domaines: str = Field(
        description=r"""name of new Domaines""", default=""
    )
    ascii: Optional[bool] = Field(
        description=r"""writing Domaines in ascii format""", default=None
    )
    single_hdf: Optional[bool] = Field(
        description=r"""writing Domaines in hdf format""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "name_of_initial_domaines": ["name_of_initial_zones"],
        "name_of_new_domaines": ["name_of_new_zones"],
        "ascii": [],
        "single_hdf": [],
    }


################################################################


class Maillerparallel(Interprete):
    r"""
    creates a parallel distributed hexaedral mesh of a parallelipipedic box. It is equivalent
    to creating a mesh with a single Pave, splitting it with Decouper and reloading it in
    parallel with Scatter. It only works in 3D at this time. It can also be used for a
    sequential computation (with all NPARTS=1)}
    """

    domain: str = Field(
        description=r"""the name of the domain to mesh (it must be an empty domain object).""",
        default="",
    )
    nb_nodes: List[int] = Field(
        description=r"""dimension defines the spatial dimension (currently only dimension=3 is supported), and nX, nY and nZ defines the total number of nodes in the mesh in each direction.""",
        default_factory=list,
    )
    splitting: List[int] = Field(
        description=r"""dimension is the spatial dimension and npartsX, npartsY and npartsZ are the number of parts created. The product of the number of parts must be equal to the number of processors used for the computation.""",
        default_factory=list,
    )
    ghost_thickness: int = Field(
        description=r"""the number of ghost cells (equivalent to the epaisseur_joint parameter of Decouper.""",
        default=0,
    )
    perio_x: Optional[bool] = Field(
        description=r"""change the splitting method to provide a valid mesh for periodic boundary conditions.""",
        default=None,
    )
    perio_y: Optional[bool] = Field(
        description=r"""change the splitting method to provide a valid mesh for periodic boundary conditions.""",
        default=None,
    )
    perio_z: Optional[bool] = Field(
        description=r"""change the splitting method to provide a valid mesh for periodic boundary conditions.""",
        default=None,
    )
    function_coord_x: Optional[str] = Field(
        description=r"""By default, the meshing algorithm creates nX nY nZ coordinates ranging between 0 and 1 (eg a unity size box). If function_coord_x} is specified, it is used to transform the [0,1] segment to the coordinates of the nodes. funcX must be a function of the x variable only.""",
        default=None,
    )
    function_coord_y: Optional[str] = Field(
        description=r"""like function_coord_x for y""", default=None
    )
    function_coord_z: Optional[str] = Field(
        description=r"""like function_coord_x for z""", default=None
    )
    file_coord_x: Optional[str] = Field(
        description=r"""Keyword to read the Nx floating point values used as nodes coordinates in the file.""",
        default=None,
    )
    file_coord_y: Optional[str] = Field(
        description=r"""idem file_coord_x for y""", default=None
    )
    file_coord_z: Optional[str] = Field(
        description=r"""idem file_coord_x for z""", default=None
    )
    boundary_xmin: Optional[str] = Field(
        description=r"""the name of the boundary at the minimum X direction. If it not provided, the default boundary names are xmin, xmax, ymin, ymax, zmin and zmax. If the mesh is periodic in a given direction, only the MIN boundary name is used, for both sides of the box.""",
        default=None,
    )
    boundary_xmax: Optional[str] = Field(description=r"""not_set""", default=None)
    boundary_ymin: Optional[str] = Field(description=r"""not_set""", default=None)
    boundary_ymax: Optional[str] = Field(description=r"""not_set""", default=None)
    boundary_zmin: Optional[str] = Field(description=r"""not_set""", default=None)
    boundary_zmax: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "domain": [],
        "nb_nodes": [],
        "splitting": [],
        "ghost_thickness": [],
        "perio_x": [],
        "perio_y": [],
        "perio_z": [],
        "function_coord_x": [],
        "function_coord_y": [],
        "function_coord_z": [],
        "file_coord_x": [],
        "file_coord_y": [],
        "file_coord_z": [],
        "boundary_xmin": [],
        "boundary_xmax": [],
        "boundary_ymin": [],
        "boundary_ymax": [],
        "boundary_zmin": [],
        "boundary_zmax": [],
    }


################################################################


class Tetraedriser(Interprete):
    r"""
    To achieve a tetrahedral mesh based on a mesh comprising blocks, the Tetraedriser
    (Tetrahedralise) interpretor is used in VEF discretization. Initial block is divided in 6
    tetrahedra: \includepng{{tetraedriser.jpeg}}{{5}}
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Tetraedriser_homogene(Tetraedriser):
    r"""
    Use the Tetraedriser_homogene (Homogeneous_Tetrahedralisation) interpretor in VEF
    discretization to mesh a block in tetrahedrals. Each block hexahedral is no longer divided
    into 6 tetrahedrals (keyword Tetraedriser (Tetrahedralise)), it is now broken down into 40
    tetrahedrals. Thus a block defined with 11 nodes in each X, Y, Z direction will contain
    10*10*10*40=40,000 tetrahedrals. This also allows problems in the mesh corners with the
    P1NC/P1iso/P1bulle or P1/P1 discretization items to be avoided. Initial block is divided
    in 40 tetrahedra: \includepng{{tetraedriserhomogene.jpeg}}{{5}}
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Precisiongeom(Interprete):
    r"""
    Class to change the way floating-point number comparison is done. By default, two numbers
    are equal if their absolute difference is smaller than 1e-10. The keyword is useful to
    modify this value. Moreover, nodes coordinates will be written in .geom files with this
    same precision.
    """

    precision: float = Field(description=r"""New value of precision.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "precision": []}


################################################################


class Rotation(Interprete):
    r"""
    Keyword to rotate the geometry of an arbitrary angle around an axis aligned with Ox, Oy or
    Oz axis.
    """

    domain_name: str = Field(
        description=r"""Name of domain to wich the transformation is applied.""",
        default="",
    )
    dir: Literal["x", "y", "z"] = Field(
        description=r"""X, Y or Z to indicate the direction of the rotation axis""",
        default="x",
    )
    coord1: float = Field(
        description=r"""coordinates of the center of rotation in the plane orthogonal to the rotation axis. These coordinates must be specified in the direct triad sense.""",
        default=0.0,
    )
    coord2: float = Field(description=r"""not_set""", default=0.0)
    angle: float = Field(description=r"""angle of rotation (in degrees)""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "domain_name": [],
        "dir": [],
        "coord1": [],
        "coord2": [],
        "angle": [],
    }


################################################################


class Bidim_axi(Interprete):
    r"""
    Keyword allowing a 2D calculation to be executed using axisymetric coordinates (R, Z). If
    this instruction is not included, calculations are carried out using Cartesian
    coordinates.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Domaine(Objet_u):
    r"""
    Keyword to create a domain.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Domaineaxi1d(Domaine):
    r"""
    1D domain
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Trianguler_h(Triangulate):
    r"""
    To achieve a triangular mesh from a mesh comprising rectangles (4 triangles per
    rectangle). Should be used in VEF discretization. Principle:

    \includepng{{triangulerh.pdf}}{{10}}
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Refine_mesh(Interprete):
    r"""
    not_set
    """

    domaine: str = Field(description=r"""not_set""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domaine": []}


################################################################


class Scatter(Interprete):
    r"""
    Class to read a partionned mesh from the files during a parallel calculation. The files
    are in binary format.
    """

    file: str = Field(description=r"""Name of file.""", default="")
    domaine: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "file": [], "domaine": []}


################################################################


class Nettoiepasnoeuds(Interprete):
    r"""
    Keyword NettoiePasNoeuds does not delete useless nodes (nodes without elements) from a
    domain.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Distance_paroi(Interprete):
    r"""
    Class to generate external file Wall_length.xyz devoted for instance, for mixing length
    modelling. In this file, are saved the coordinates of each element (center of gravity) of
    dom domain and minimum distance between this point and boundaries (specified bords) that
    user specifies in data file (typically, those associated to walls). A field Distance_paroi
    is available to post process the distance to the wall.
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    bords: List[str] = Field(description=r"""Boundaries.""", default_factory=list)
    format: Literal["binaire", "formatte"] = Field(
        description=r"""Value for format may be binaire (a binary file Wall_length.xyz is written) or formatte (moreover, a formatted file Wall_length_formatted.xyz is written).""",
        default="binaire",
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bords": [], "format": []}


################################################################


class Bord_base(Objet_lecture):
    r"""
    Basic class for block sides. Block sides that are neither edges nor connectors are not
    specified. The duplicate nodes of two blocks in contact are automatically recognized and
    deleted.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Defbord(Objet_lecture):
    r"""
    Class to define an edge.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Internes(Bord_base):
    r"""
    To indicate that the block has a set of internal faces (these faces will be duplicated
    automatically by the program and will be processed in a manner similar to edge faces).

    Two boundaries with the same boundary conditions may have the same name (whether or not
    they belong to the same block).

    The keyword Internes (Internal) must be used to execute a calculation with plates,
    followed by the equation of the surface area covered by the plates.
    """

    nom: str = Field(description=r"""Name of block side.""", default="")
    defbord: Defbord = Field(
        description=r"""Definition of block side.""",
        default_factory=lambda: eval("Defbord()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nom": [], "defbord": []}


################################################################


class Corriger_frontiere_periodique(Interprete):
    r"""
    The Corriger_frontiere_periodique keyword is mandatory to first define the periodic
    boundaries, to reorder the faces and eventually fix unaligned nodes of these boundaries.
    Faces on one side of the periodic domain are put first, then the faces on the opposite
    side, in the same order. It must be run in sequential before mesh splitting.
    """

    domaine: str = Field(description=r"""Name of domain.""", default="")
    bord: str = Field(
        description=r"""the name of the boundary (which must contain two opposite sides of the domain)""",
        default="",
    )
    direction: Optional[List[float]] = Field(
        description=r"""defines the periodicity direction vector (a vector that points from one node on one side to the opposite node on the other side). This vector must be given if the automatic algorithm fails, that is: - when the node coordinates are not perfectly periodic  - when the periodic direction is not aligned with the normal vector of the boundary faces""",
        default=None,
    )
    fichier_post: Optional[str] = Field(description=r""".""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "bord": [],
        "direction": [],
        "fichier_post": [],
    }


################################################################


class List_bord(Listobj):
    r"""
    The block sides.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Extruder_en20(Interprete):
    r"""
    It does the same task as Extruder except that a prism is cut into 20 tetraedra instead of
    3. The name of the boundaries will be devant (front) and derriere (back). But you can
    change these names with the keyword RegroupeBord.
    """

    domaine: str = Field(description=r"""Name of the domain.""", default="")
    nb_tranches: int = Field(
        description=r"""Number of elements in the extrusion direction.""", default=0
    )
    direction: Optional[Troisf] = Field(
        description=r"""0 Direction of the extrude operation.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": ["domain_name"],
        "nb_tranches": [],
        "direction": [],
    }


################################################################


class Extraire_surface(Interprete):
    r"""
    This keyword extracts a surface mesh named domain_name (this domain should have been
    declared before) from the mesh of the pb_name problem. The surface mesh is defined by one
    or two conditions. The first condition is about elements with Condition_elements. For
    example: Condition_elements x*x+y*y+z*z<1

    Will define a surface mesh with external faces of the mesh elements inside the sphere of
    radius 1 located at (0,0,0). The second condition Condition_faces is useful to give a
    restriction.

    By default, the faces from the boundaries are not added to the surface mesh excepted if
    option avec_les_bords is given (all the boundaries are added), or if the option
    avec_certains_bords is used to add only some boundaries.
    """

    domaine: str = Field(description=r"""Domain in which faces are saved""", default="")
    probleme: str = Field(
        description=r"""Problem from which faces should be extracted""", default=""
    )
    condition_elements: Optional[str] = Field(
        description=r"""condition on center of elements""", default=None
    )
    condition_faces: Optional[str] = Field(description=r"""not_set""", default=None)
    avec_les_bords: Optional[bool] = Field(description=r"""not_set""", default=None)
    avec_certains_bords: Optional[List[str]] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "probleme": [],
        "condition_elements": [],
        "condition_faces": [],
        "avec_les_bords": [],
        "avec_certains_bords": [],
    }


################################################################


class Decouper_bord_coincident(Interprete):
    r"""
    In case of non-coincident meshes and a paroi_contact condition, run is stopped and two
    external files are automatically generated in VEF (connectivity_failed_boundary_name and
    connectivity_failed_pb_name.med). In 2D, the keyword Decouper_bord_coincident associated
    to the connectivity_failed_boundary_name file allows to generate a new coincident mesh.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    bord: str = Field(description=r"""connectivity_failed_boundary_name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "bord": []}


################################################################


class Orientefacesbord(Interprete):
    r"""
    Keyword to modify the order of the boundary vertices included in a domain, such that the
    surface normals are outer pointing.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Raccord(Bord_base):
    r"""
    The block side is in contact with the block of another domain (case of two coupled
    problems).
    """

    type1: Literal["local", "distant"] = Field(
        description=r"""Contact type.""", default="local"
    )
    type2: Literal["homogene"] = Field(
        description=r"""Contact type.""", default="homogene"
    )
    nom: str = Field(description=r"""Name of block side.""", default="")
    defbord: Defbord = Field(
        description=r"""Definition of block side.""",
        default_factory=lambda: eval("Defbord()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "type1": [],
        "type2": [],
        "nom": [],
        "defbord": [],
    }


################################################################


class Extraire_domaine(Interprete):
    r"""
    Keyword to create a new domain built with the domain elements of the pb_name problem
    verifying the two conditions given by Condition_elements. The problem pb_name should have
    been discretized.
    """

    domaine: str = Field(description=r"""Domain in which faces are saved""", default="")
    probleme: str = Field(
        description=r"""Problem from which faces should be extracted""", default=""
    )
    condition_elements: Optional[str] = Field(description=r"""not_set""", default=None)
    sous_domaine: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "probleme": [],
        "condition_elements": [],
        "sous_domaine": ["sous_zone"],
    }


################################################################


class Verifier_simplexes(Interprete):
    r"""
    Keyword to raffine a simplexes
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Lire_ideas(Interprete):
    r"""
    Read a geom in a unv file. 3D tetra mesh elements only may be read by TRUST.
    """

    nom_dom: str = Field(description=r"""Name of domain.""", default="")
    file: str = Field(description=r"""Name of file.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "file": []}


################################################################


class Regroupebord(Interprete):
    r"""
    Keyword to build one boundary new_bord with several boundaries of the domain named
    domaine.
    """

    domaine: str = Field(description=r"""Name of domain""", default="")
    new_bord: str = Field(description=r"""Name of the new boundary""", default="")
    bords: Bloc_lecture = Field(
        description=r"""{ Bound1 Bound2 }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": ["domain"],
        "new_bord": [],
        "bords": [],
    }


################################################################


class Extraire_plan(Interprete):
    r"""
    This keyword extracts a plane mesh named domain_name (this domain should have been
    declared before) from the mesh of the pb_name problem. The plane can be either a triangle
    (defined by the keywords Origine, Point1, Point2 and Triangle), either a regular
    quadrangle (with keywords Origine, Point1 and Point2), or either a generalized quadrangle
    (with keywords Origine, Point1, Point2, Point3). The keyword Epaisseur specifies the
    thickness of volume around the plane which contains the faces of the extracted mesh. The
    keyword via_extraire_surface will create a plan and use Extraire_surface algorithm.
    Inverse_condition_element keyword then will be used in the case where the plane is a
    boundary not well oriented, and avec_certains_bords_pour_extraire_surface is the option
    related to the Extraire_surface option named avec_certains_bords.
    """

    domaine: str = Field(description=r"""domain name""", default="")
    probleme: str = Field(description=r"""pb_name""", default="")
    origine: List[float] = Field(description=r"""not_set""", default_factory=list)
    point1: List[float] = Field(description=r"""not_set""", default_factory=list)
    point2: List[float] = Field(description=r"""not_set""", default_factory=list)
    point3: Optional[List[float]] = Field(description=r"""not_set""", default=None)
    triangle: Optional[bool] = Field(description=r"""not_set""", default=None)
    epaisseur: float = Field(description=r"""thickness""", default=0.0)
    via_extraire_surface: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    inverse_condition_element: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    avec_certains_bords_pour_extraire_surface: Optional[List[str]] = Field(
        description=r"""name of boundaries to include when extracting plan""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "probleme": [],
        "origine": [],
        "point1": [],
        "point2": [],
        "point3": [],
        "triangle": [],
        "epaisseur": [],
        "via_extraire_surface": [],
        "inverse_condition_element": [],
        "avec_certains_bords_pour_extraire_surface": [],
    }


################################################################


class Dilate(Interprete):
    r"""
    Keyword to multiply the whole coordinates of the geometry.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    alpha: float = Field(
        description=r"""Value of dilatation coefficient.""", default=0.0
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "alpha": []}


################################################################


class Remove_invalid_internal_boundaries(Interprete):
    r"""
    Keyword to suppress an internal boundary of the domain_name domain. Indeed, some mesh
    tools may define internal boundaries (eg: for post processing task after the calculation)
    but TRUST does not support it yet.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Reorienter_tetraedres(Interprete):
    r"""
    This keyword is mandatory for front-tracking computations with the VEF discretization. For
    each tetrahedral element of the domain, it checks if it has a positive volume. If the
    volume (determinant of the three vectors) is negative, it swaps two nodes to reverse the
    orientation of this tetrahedron.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Tetraedriser_homogene_compact(Tetraedriser):
    r"""
    This new discretization generates tetrahedral elements from cartesian or non-cartesian
    hexahedral elements. The process cut each hexahedral in 6 pyramids, each of them being cut
    then in 4 tetrahedral. So, in comparison with tetra_homogene, less elements (*24 instead
    of*40) with more homogeneous volumes are generated. Moreover, this process is done in a
    faster way. Initial block is divided in 24 tetrahedra:
    \includepng{{tetraedriserhomogenecompact.jpeg}}{{5}}
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Dimension(Interprete):
    r"""
    Keyword allowing calculation dimensions to be set (2D or 3D), where dim is an integer set
    to 2 or 3. This instruction is mandatory.
    """

    dim: Literal[2, 3] = Field(description=r"""Number of dimensions.""", default=2)
    _synonyms: ClassVar[dict] = {None: [], "dim": []}


################################################################


class Tetraedriser_homogene_fin(Tetraedriser):
    r"""
    Tetraedriser_homogene_fin is the recommended option to tetrahedralise blocks. As an
    extension (subdivision) of Tetraedriser_homogene_compact, this last one cut each initial
    block in 48 tetrahedra (against 24, previously). This cutting ensures :

    - a correct cutting in the corners (in respect to pressure discretization PreP1B),

    - a better isotropy of elements than with Tetraedriser_homogene_compact,

    - a better alignment of summits (this could have a benefit effect on calculation near
    walls since first elements in contact with it are all contained in the same constant
    thickness and ii/ by the way, a 3D cartesian grid based on summits can be engendered and
    used to realise spectral analysis in HIT for instance). Initial block is divided in 48
    tetrahedra: \includepng{{tetraedriserhomogenefin.jpeg}}{{5}}
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Resequencing(Interprete):
    r"""
    The Reordonner_32_64 interpretor is required sometimes for a VDF mesh which is not
    produced by the internal mesher. Example where this is used:

    Read_file dom fichier.geom

    Reordonner_32_64 dom

    Observations: This keyword is redundant when the mesh that is read is correctly sequenced
    in the TRUST sense. This significant mesh operation may take some time... The message
    returned by TRUST is not explicit when the Reordonner_32_64 (Resequencing) keyword is
    required but not included in the data set...
    """

    domain_name: str = Field(
        description=r"""Name of domain to resequence.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: ["reordonner"], "domain_name": []}


################################################################


class Supprime_bord(Interprete):
    r"""
    Keyword to remove boundaries (named Boundary_name1 Boundary_name2 ) of the domain named
    domain_name.
    """

    domaine: str = Field(description=r"""Name of domain""", default="")
    bords: Annotated[List[Nom_anonyme], "List_nom"] = Field(
        description=r"""List of name.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "domaine": ["domain"], "bords": []}


################################################################


class Raffiner_isotrope(Interprete):
    r"""
    For VDF and VEF discretizations, allows to cut triangles/quadrangles or
    tetrahedral/hexaedras elements respectively in 4 or 8 new ones by defining new summits
    located at the middle of edges (and center of faces and elements for quadrangles and
    hexaedra). Such a cut preserves the shape of original elements (isotropic). For 2D
    elements: \includepng{{raffinerisotrirect.jpeg}}{{6}}

    For 3D elements: \includepng{{raffinerisotetra.jpeg}}{{6}}

    \includepng{{raffinerisohexa.jpeg}}{{5}}.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: ["raffiner_simplexes"], "domain_name": []}


################################################################


class Extrudeparoi(Interprete):
    r"""
    Keyword dedicated in 3D (VEF) to create prismatic layer at wall. Each prism is cut into 3
    tetraedra.
    """

    domaine: str = Field(description=r"""Name of the domain.""", default="")
    nom_bord: str = Field(
        description=r"""Name of the (no-slip) boundary for creation of prismatic layers.""",
        default="",
    )
    epaisseur: Optional[List[float]] = Field(
        description=r"""n r1 r2 .... rn : (relative or absolute) width for each layer.""",
        default=None,
    )
    critere_absolu: Optional[int] = Field(
        description=r"""relative (0, the default) or absolute (1) width for each layer.""",
        default=None,
    )
    projection_normale_bord: Optional[bool] = Field(
        description=r"""keyword to project layers on the same plane that contiguous boundaries. defaut values are : epaisseur_relative 1 0.5 projection_normale_bord 1""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "nom_bord": [],
        "epaisseur": [],
        "critere_absolu": [],
        "projection_normale_bord": [],
    }


################################################################


class Verifier_qualite_raffinements(Interprete):
    r"""
    not_set
    """

    domain_names: Annotated[List[Nom_anonyme], "Vect_nom"] = Field(
        description=r"""Vect of name.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_names": []}


################################################################


class Extrudebord(Interprete):
    r"""
    Class to generate an extruded mesh from a boundary of a tetrahedral or an hexahedral mesh.

    Warning: If the initial domain is a tetrahedral mesh, the boundary will be moved in the
    XY plane then extrusion will be applied (you should maybe use the Transformer keyword on
    the final domain to have the domain you really want). You can use the keyword
    Postraiter_domaine to generate a lata|med|... file to visualize your initial and final
    meshes.

    This keyword can be used for example to create a periodic box extracted from a boundary
    of a tetrahedral or a hexaedral mesh. This periodic box may be used then to engender
    turbulent inlet flow condition for the main domain.

    Note that ExtrudeBord in VEF generates 3 or 14 tetrahedra from extruded prisms.
    """

    domaine_init: str = Field(
        description=r"""Initial domain with hexaedras or tetrahedras.""", default=""
    )
    direction: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Directions for the extrusion.""", default_factory=list
    )
    nb_tranches: int = Field(
        description=r"""Number of elements in the extrusion direction.""", default=0
    )
    domaine_final: str = Field(description=r"""Extruded domain.""", default="")
    nom_bord: str = Field(
        description=r"""Name of the boundary of the initial domain where extrusion will be applied.""",
        default="",
    )
    hexa_old: Optional[bool] = Field(
        description=r"""Old algorithm for boundary extrusion from a hexahedral mesh.""",
        default=None,
    )
    trois_tetra: Optional[bool] = Field(
        description=r"""To extrude in 3 tetrahedras instead of 14 tetrahedras.""",
        default=None,
    )
    vingt_tetra: Optional[bool] = Field(
        description=r"""To extrude in 20 tetrahedras instead of 14 tetrahedras.""",
        default=None,
    )
    sans_passer_par_le2d: Optional[int] = Field(
        description=r"""Only for non-regression""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine_init": [],
        "direction": [],
        "nb_tranches": [],
        "domaine_final": [],
        "nom_bord": [],
        "hexa_old": [],
        "trois_tetra": [],
        "vingt_tetra": [],
        "sans_passer_par_le2d": [],
    }


################################################################


class Read_tgrid(Interprete):
    r"""
    Keyword to reaf Tgrid/Gambit mesh files. 2D (triangles or quadrangles) and 3D (tetra or
    hexa elements) meshes, may be read by TRUST.
    """

    dom: str = Field(description=r"""Name of domaine.""", default="")
    filename: str = Field(
        description=r"""Name of file containing the mesh.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: ["lire_tgrid"], "dom": [], "filename": []}


################################################################


class Interprete_geometrique_base(Interprete):
    r"""
    Class for interpreting a data file
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Rectify_mesh(Interprete):
    r"""
    Keyword to raffine a mesh
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: ["orienter_simplexes"], "domain_name": []}


################################################################


class Analyse_angle(Interprete):
    r"""
    Keyword Analyse_angle prints the histogram of the largest angle of each mesh elements of
    the domain named name_domain. nb_histo is the histogram number of bins. It is called by
    default during the domain discretization with nb_histo set to 18. Useful to check the
    number of elements with angles above 90 degrees.
    """

    domain_name: str = Field(
        description=r"""Name of domain to resequence.""", default=""
    )
    nb_histo: int = Field(description=r"""not_set""", default=0)
    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "nb_histo": []}


################################################################


class Create_domain_from_sub_domain(Interprete_geometrique_base):
    r"""
    This keyword fills the domain domaine_final with the subdomaine par_sous_zone from the
    domain domaine_init. It is very useful when meshing several mediums with Gmsh. Each medium
    will be defined as a subdomaine into Gmsh. A MED mesh file will be saved from Gmsh and
    read with Lire_Med keyword by the TRUST data file. And with this keyword, a domain will be
    created for each medium in the TRUST data file.
    """

    domaine_final: Optional[str] = Field(
        description=r"""new domain in which faces are stored""", default=None
    )
    par_sous_zone: Optional[str] = Field(
        description=r"""a sub-area (a group in a MED file) allowing to choose the elements""",
        default=None,
    )
    domaine_init: str = Field(description=r"""initial domain""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["create_domain_from_sub_domains", "create_domain_from_sous_zone"],
        "domaine_final": [],
        "par_sous_zone": ["par_sous_dom"],
        "domaine_init": [],
    }


################################################################


class Lecture_bloc_moment_base(Objet_lecture):
    r"""
    Auxiliary class to compute and print the moments.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Calculer_moments(Interprete):
    r"""
    Calculates and prints the torque (moment of force) exerted by the fluid on each boundary
    in output files (.out) of the domain nom_dom.
    """

    nom_dom: str = Field(description=r"""Name of domain.""", default="")
    mot: Lecture_bloc_moment_base = Field(
        description=r"""Keyword.""",
        default_factory=lambda: eval("Lecture_bloc_moment_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "mot": []}


################################################################


class Calcul(Lecture_bloc_moment_base):
    r"""
    The centre of gravity will be calculated.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Centre_de_gravite(Lecture_bloc_moment_base):
    r"""
    To specify the centre of gravity.
    """

    point: Un_point = Field(
        description=r"""A centre of gravity.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "point": []}


################################################################


class Defbord_2(Defbord):
    r"""
    1-D edge (straight line) in the 2-D space.
    """

    dir: Literal["x", "y"] = Field(
        description=r"""Edge is perpendicular to this direction.""", default="x"
    )
    eq: Literal["="] = Field(description=r"""Equality sign.""", default="=")
    pos: float = Field(description=r"""Position value.""", default=0.0)
    pos2_min: float = Field(description=r"""Minimal value.""", default=0.0)
    inf1: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    dir2: Literal["x", "y"] = Field(
        description=r"""Edge is parallel to this direction.""", default="x"
    )
    inf2: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    pos2_max: float = Field(description=r"""Maximal value.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "dir": [],
        "eq": [],
        "pos": [],
        "pos2_min": [],
        "inf1": [],
        "dir2": [],
        "inf2": [],
        "pos2_max": [],
    }


################################################################


class Defbord_3(Defbord):
    r"""
    2-D edge (plane) in the 3-D space.
    """

    dir: Literal["x", "y", "z"] = Field(
        description=r"""Edge is perpendicular to this direction.""", default="x"
    )
    eq: Literal["="] = Field(description=r"""Equality sign.""", default="=")
    pos: float = Field(description=r"""Position value.""", default=0.0)
    pos2_min: float = Field(description=r"""Minimal value.""", default=0.0)
    inf1: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    dir2: Literal["x", "y"] = Field(
        description=r"""Edge is parallel to this direction.""", default="x"
    )
    inf2: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    pos2_max: float = Field(description=r"""Maximal value.""", default=0.0)
    pos3_min: float = Field(description=r"""Minimal value.""", default=0.0)
    inf3: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    dir3: Literal["y", "z"] = Field(
        description=r"""Edge is parallel to this direction.""", default="y"
    )
    inf4: Literal["<="] = Field(
        description=r"""Less than or equal to sign.""", default="<="
    )
    pos3_max: float = Field(description=r"""Maximal value.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "dir": [],
        "eq": [],
        "pos": [],
        "pos2_min": [],
        "inf1": [],
        "dir2": [],
        "inf2": [],
        "pos2_max": [],
        "pos3_min": [],
        "inf3": [],
        "dir3": [],
        "inf4": [],
        "pos3_max": [],
    }


################################################################


class Bord(Bord_base):
    r"""
    The block side is not in contact with another block and boundary conditions are applied to
    it.
    """

    nom: str = Field(description=r"""Name of block side.""", default="")
    defbord: Defbord = Field(
        description=r"""Definition of block side.""",
        default_factory=lambda: eval("Defbord()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nom": [], "defbord": []}


################################################################


class Raffiner_anisotrope(Interprete):
    r"""
    Only for VEF discretizations, allows to cut triangle elements in 3, or tetrahedra in 4
    parts, by defining a new summit located at the center of the element:
    \includepng{{raffineranisotri.pdf}}{{4}} \includepng{{raffineranisotetra.jpeg}}{{6}}

    Note that such a cut creates flat elements (anisotropic).
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Modifydomaineaxi1d(Interprete):
    r"""
    Convert a 1D mesh to 1D axisymmetric mesh
    """

    dom: str = Field(description=r"""not_set""", default="")
    bloc: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: ["convert_1d_to_1daxi"], "dom": [], "bloc": []}


################################################################


class Modif_bord_to_raccord(Interprete):
    r"""
    Keyword to convert a boundary of domain_name domain of kind Bord to a boundary of kind
    Raccord (named boundary_name). It is useful when using meshes with boundaries of kind Bord
    defined and to run a coupled calculation.
    """

    domaine: str = Field(description=r"""Name of domain""", default="")
    nom_bord: str = Field(
        description=r"""Name of the boundary to transform.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: [], "domaine": ["domain"], "nom_bord": []}


################################################################


class Bloc_origine_cotes(Objet_lecture):
    r"""
    Class to create a rectangle (or a box).
    """

    name: Literal["origine"] = Field(
        description=r"""Keyword to define the origin of the rectangle (or the box).""",
        default="origine",
    )
    origin: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Coordinates of the origin of the rectangle (or the box).""",
        default_factory=list,
    )
    name2: Literal["cotes"] = Field(
        description=r"""Keyword to define the length along the axes.""", default="cotes"
    )
    cotes: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Length along the axes.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "name": [],
        "origin": ["origine"],
        "name2": [],
        "cotes": [],
    }


################################################################


class Bloc_couronne(Objet_lecture):
    r"""
    Class to create a couronne (2D).
    """

    name: Literal["origine"] = Field(
        description=r"""Keyword to define the center of the circle.""",
        default="origine",
    )
    origin: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Center of the circle.""", default_factory=list
    )
    name3: Literal["ri"] = Field(
        description=r"""Keyword to define the interior radius.""", default="ri"
    )
    ri: float = Field(description=r"""Interior radius.""", default=0.0)
    name4: Literal["re"] = Field(
        description=r"""Keyword to define the exterior radius.""", default="re"
    )
    re: float = Field(description=r"""Exterior radius.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "name": [],
        "origin": ["origine"],
        "name3": [],
        "ri": [],
        "name4": [],
        "re": [],
    }


################################################################


class Bloc_tube(Objet_lecture):
    r"""
    Class to create a tube (3D).
    """

    name: Literal["origine"] = Field(
        description=r"""Keyword to define the center of the tube.""", default="origine"
    )
    origin: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""Center of the tube.""", default_factory=list
    )
    name2: Literal["dir"] = Field(
        description=r"""Keyword to define the direction of the main axis.""",
        default="dir",
    )
    direction: Literal["x", "y", "z"] = Field(
        description=r"""direction of the main axis X, Y or Z""", default="x"
    )
    name3: Literal["ri"] = Field(
        description=r"""Keyword to define the interior radius.""", default="ri"
    )
    ri: float = Field(description=r"""Interior radius.""", default=0.0)
    name4: Literal["re"] = Field(
        description=r"""Keyword to define the exterior radius.""", default="re"
    )
    re: float = Field(description=r"""Exterior radius.""", default=0.0)
    name5: Literal["hauteur"] = Field(
        description=r"""Keyword to define the heigth of the tube.""", default="hauteur"
    )
    h: float = Field(description=r"""Heigth of the tube.""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "name": [],
        "origin": ["origine"],
        "name2": [],
        "direction": [],
        "name3": [],
        "ri": [],
        "name4": [],
        "re": [],
        "name5": [],
        "h": [],
    }


################################################################


class Sous_zone(Objet_u):
    r"""
    It is an object type describing a domain sub-set.

    A Sous_Zone (Sub-area) type object must be associated with a Domaine type object. The
    Read (Lire) interpretor is used to define the items comprising the sub-area.

    Caution: The Domain type object nom_domaine must have been meshed (and triangulated or
    tetrahedralised in VEF) prior to carrying out the Associate (Associer) nom_sous_zone
    nom_domaine instruction; this instruction must always be preceded by the read instruction.
    """

    restriction: Optional[str] = Field(
        description=r"""The elements of the sub-area nom_sous_zone must be included into the other sub-area named nom_sous_zone2. This keyword should be used first in the Read keyword.""",
        default=None,
    )
    rectangle: Optional[Bloc_origine_cotes] = Field(
        description=r"""The sub-area will include all the domain elements whose centre of gravity is within the Rectangle (in dimension 2).""",
        default=None,
    )
    segment: Optional[Bloc_origine_cotes] = Field(
        description=r"""not_set""", default=None
    )
    boite: Optional[Bloc_origine_cotes] = Field(
        description=r"""The sub-area will include all the domain elements whose centre of gravity is within the Box (in dimension 3).""",
        default=None,
    )
    liste: Optional[List[int]] = Field(
        description=r"""The sub-area will include n domain items, numbers No. 1 No. i No. n.""",
        default=None,
    )
    fichier: Optional[str] = Field(
        description=r"""The sub-area is read into the file filename.""", default=None
    )
    intervalle: Optional[Deuxentiers] = Field(
        description=r"""The sub-area will include domain items whose number is between n1 and n2 (where n1<=n2).""",
        default=None,
    )
    polynomes: Optional[Bloc_lecture] = Field(
        description=r"""A REPRENDRE""", default=None
    )
    couronne: Optional[Bloc_couronne] = Field(
        description=r"""In 2D case, to create a couronne.""", default=None
    )
    tube: Optional[Bloc_tube] = Field(
        description=r"""In 3D case, to create a tube.""", default=None
    )
    fonction_sous_zone: Optional[str] = Field(
        description=r"""Keyword to build a sub-area with the the elements included into the area defined by fonction>0.""",
        default=None,
    )
    union: Optional[str] = Field(
        description=r"""The elements of the sub-area nom_sous_zone3 will be added to the sub-area nom_sous_zone. This keyword should be used last in the Read keyword.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["sous_domaine"],
        "restriction": [],
        "rectangle": [],
        "segment": [],
        "boite": ["box"],
        "liste": [],
        "fichier": ["filename"],
        "intervalle": [],
        "polynomes": [],
        "couronne": [],
        "tube": [],
        "fonction_sous_zone": ["fonction_sous_domaine"],
        "union": ["union_with"],
    }


################################################################


class Polyedriser(Interprete):
    r"""
    cast hexahedra into polyhedra so that the indexing of the mesh vertices is compatible with
    PolyMAC_P0P1NC discretization. Must be used in PolyMAC_P0P1NC discretization if a
    hexahedral mesh has been produced with TRUST's internal mesh generator.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Tetraedriser_par_prisme(Tetraedriser):
    r"""
    Tetraedriser_par_prisme generates 6 iso-volume tetrahedral element from primary hexahedral
    one (contrarily to the 5 elements ordinarily generated by tetraedriser). This element is
    suitable for calculation of gradients at the summit (coincident with the gravity centre of
    the jointed elements related with) and spectra (due to a better alignment of the points).
    \includepng{{tetraedriserparprisme.jpeg}}{{5}}

    \includepng{{tetraedriserparprisme2.jpeg}}{{5}}

    Initial block is divided in 6 prismes.
    """

    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Axi(Interprete):
    r"""
    This keyword allows a 3D calculation to be executed using cylindrical coordinates
    (R,$\jolitheta$,Z). If this instruction is not included, calculations are carried out
    using Cartesian coordinates.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Reorienter_triangles(Interprete):
    r"""
    not_set
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Discretiser_domaine(Interprete):
    r"""
    Useful to discretize the domain domain_name (faces will be created) without defining a
    problem.
    """

    domain_name: str = Field(description=r"""Name of the domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Transformer(Interprete):
    r"""
    Keyword to transform the coordinates of the geometry.

    Exemple to rotate your mesh by a 90o rotation and to scale the z coordinates by a factor
    2: Transformer domain_name -y -x 2*z
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    formule: Annotated[List[str], "size_is_dim"] = Field(
        description=r"""Function_for_x Function_for_y \[ Function_for z \]""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_name": [], "formule": []}


################################################################


class List_bloc_mailler(Listobj):
    r"""
    List of block mesh.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Mailler_base(Objet_lecture):
    r"""
    Basic class to mesh.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Mailler(Interprete):
    r"""
    The Mailler (Mesh) interpretor allows a Domain type object domaine to be meshed with
    objects objet_1, objet_2, etc...
    """

    domaine: str = Field(description=r"""Name of domain.""", default="")
    bloc: Annotated[List[Mailler_base], "List_bloc_mailler"] = Field(
        description=r"""List of block mesh.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "domaine": [], "bloc": []}


################################################################


class Bloc_pave(Objet_lecture):
    r"""
    Class to create a pave.
    """

    origine: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""Keyword to define the pave (block) origin, that is to say one of the 8 block points (or 4 in a 2D coordinate system).""",
        default=None,
    )
    longueurs: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""Keyword to define the block dimensions, that is to say knowing the origin, length along the axes.""",
        default=None,
    )
    nombre_de_noeuds: Optional[Annotated[List[int], "size_is_dim"]] = Field(
        description=r"""Keyword to define the discretization (nodenumber) in each direction.""",
        default=None,
    )
    facteurs: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""Keyword to define stretching factors for mesh discretization in each direction. This is a real number which must be positive (by default 1.0). A stretching factor other than 1 allows refinement on one edge in one direction.""",
        default=None,
    )
    symx: Optional[bool] = Field(
        description=r"""Keyword to define a block mesh that is symmetrical with respect to the YZ plane (respectively Y-axis in 2D) passing through the block centre.""",
        default=None,
    )
    symy: Optional[bool] = Field(
        description=r"""Keyword to define a block mesh that is symmetrical with respect to the XZ plane (respectively X-axis in 2D) passing through the block centre.""",
        default=None,
    )
    symz: Optional[bool] = Field(
        description=r"""Keyword defining a block mesh that is symmetrical with respect to the XY plane passing through the block centre.""",
        default=None,
    )
    xtanh: Optional[float] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the X-direction.""",
        default=None,
    )
    xtanh_dilatation: Optional[Literal[-1, 0, 1]] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the X-direction. xtanh_dilatation: The value may be -1,0,1 (0 by default): 0: coarse mesh at the middle of the channel and smaller near the walls -1: coarse mesh at the left side of the channel and smaller at the right side 1: coarse mesh at the right side of the channel and smaller near the left side of the channel.""",
        default=None,
    )
    xtanh_taille_premiere_maille: Optional[float] = Field(
        description=r"""Size of the first cell of the mesh with tanh (hyperbolic tangent) variation in the X-direction.""",
        default=None,
    )
    ytanh: Optional[float] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the Y-direction.""",
        default=None,
    )
    ytanh_dilatation: Optional[Literal[-1, 0, 1]] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the Y-direction. ytanh_dilatation: The value may be -1,0,1 (0 by default): 0: coarse mesh at the middle of the channel and smaller near the walls -1: coarse mesh at the bottom of the channel and smaller near the top 1: coarse mesh at the top of the channel and smaller near the bottom.""",
        default=None,
    )
    ytanh_taille_premiere_maille: Optional[float] = Field(
        description=r"""Size of the first cell of the mesh with tanh (hyperbolic tangent) variation in the Y-direction.""",
        default=None,
    )
    ztanh: Optional[float] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the Z-direction.""",
        default=None,
    )
    ztanh_dilatation: Optional[Literal[-1, 0, 1]] = Field(
        description=r"""Keyword to generate mesh with tanh (hyperbolic tangent) variation in the Z-direction. tanh_dilatation: The value may be -1,0,1 (0 by default): 0: coarse mesh at the middle of the channel and smaller near the walls -1: coarse mesh at the back of the channel and smaller near the front 1: coarse mesh at the front of the channel and smaller near the back.""",
        default=None,
    )
    ztanh_taille_premiere_maille: Optional[float] = Field(
        description=r"""Size of the first cell of the mesh with tanh (hyperbolic tangent) variation in the Z-direction.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "origine": [],
        "longueurs": [],
        "nombre_de_noeuds": [],
        "facteurs": [],
        "symx": [],
        "symy": [],
        "symz": [],
        "xtanh": [],
        "xtanh_dilatation": [],
        "xtanh_taille_premiere_maille": [],
        "ytanh": [],
        "ytanh_dilatation": [],
        "ytanh_taille_premiere_maille": [],
        "ztanh": [],
        "ztanh_dilatation": [],
        "ztanh_taille_premiere_maille": [],
    }


################################################################


class Pave(Mailler_base):
    r"""
    Class to create a pave (block) with boundaries.
    """

    name: str = Field(description=r"""Name of the pave (block).""", default="")
    bloc: Bloc_pave = Field(
        description=r"""Definition of the pave (block).""",
        default_factory=lambda: eval("Bloc_pave()"),
    )
    list_bord: Annotated[List[Bord_base], "List_bord"] = Field(
        description=r"""The block sides.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "name": [], "bloc": [], "list_bord": []}


################################################################


class Epsilon(Mailler_base):
    r"""
    Two points will be confused if the distance between them is less than eps. By default, eps
    is set to 1e-12. The keyword Epsilon allows an alternative value to be assigned to eps.
    """

    eps: float = Field(description=r"""New value of precision.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "eps": []}


################################################################


class Domain(Mailler_base):
    r"""
    Class to reuse a domain.
    """

    domain_name: str = Field(description=r"""Name of domain.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Redresser_hexaedres_vdf(Interprete):
    r"""
    Keyword to convert a domain (named domain_name) with quadrilaterals/VEF hexaedras which
    looks like rectangles/VDF hexaedras into a domain with real rectangles/VDF hexaedras.
    """

    domain_name: str = Field(
        description=r"""Name of domain to resequence.""", default=""
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_name": []}


################################################################


class Interpolation_ibm_base(Objet_u):
    r"""
    Base class for all the interpolation methods available in the Immersed Boundary Method
    (IBM).
    """

    impr: Optional[bool] = Field(
        description=r"""To print IBM-related data""", default=None
    )
    nb_histo_boxes_impr: Optional[int] = Field(
        description=r"""number of histogram boxes for printed data""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "impr": [], "nb_histo_boxes_impr": []}


################################################################


class Interpolation_ibm_elem_fluid(Interpolation_ibm_base):
    r"""
    Immersed Boundary Method (IBM): fluid element interpolation.
    """

    points_fluides: Field_base = Field(
        description=r"""Node field giving the projection of the point below (points_solides) falling into the pure cell fluid""",
        default_factory=lambda: eval("Field_base()"),
    )
    points_solides: Field_base = Field(
        description=r"""Node field giving the projection of the node on the immersed boundary""",
        default_factory=lambda: eval("Field_base()"),
    )
    elements_fluides: Field_base = Field(
        description=r"""Node field giving the number of the element (cell) containing the pure fluid point""",
        default_factory=lambda: eval("Field_base()"),
    )
    correspondance_elements: Field_base = Field(
        description=r"""Cell field giving the SALOME cell number""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["interpolation_ibm_element_fluide", "ibm_element_fluide"],
        "points_fluides": [],
        "points_solides": [],
        "elements_fluides": [],
        "correspondance_elements": [],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Interpolation_ibm_power_law_tbl(Interpolation_ibm_elem_fluid):
    r"""
    Immersed Boundary Method (IBM): power law interpolation.
    """

    formulation_linear_pwl: Optional[int] = Field(
        description=r"""Choix formulation lineaire ou non""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["ibm_power_law_tbl"],
        "formulation_linear_pwl": [],
        "points_fluides": [],
        "points_solides": [],
        "elements_fluides": [],
        "correspondance_elements": [],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Interpolation_ibm_hybride(Interpolation_ibm_elem_fluid):
    r"""
    Immersed Boundary Method (IBM): hybrid (fluid/mean gradient) interpolation.
    """

    est_dirichlet: Field_base = Field(
        description=r"""Node field of booleans indicating whether the node belong to an element where the interface is""",
        default_factory=lambda: eval("Field_base()"),
    )
    elements_solides: Field_base = Field(
        description=r"""Node field giving the element number containing the solid point""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["ibm_hybride"],
        "est_dirichlet": [],
        "elements_solides": [],
        "points_fluides": [],
        "points_solides": [],
        "elements_fluides": [],
        "correspondance_elements": [],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Interpolation_ibm_aucune(Interpolation_ibm_base):
    r"""
    Immersed Boundary Method (IBM): no interpolation.
    """

    _synonyms: ClassVar[dict] = {
        None: ["ibm_aucune"],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Interpolation_ibm_mean_gradient(Interpolation_ibm_base):
    r"""
    Immersed Boundary Method (IBM): mean gradient interpolation.
    """

    points_solides: Field_base = Field(
        description=r"""Node field giving the projection of the node on the immersed boundary""",
        default_factory=lambda: eval("Field_base()"),
    )
    est_dirichlet: Field_base = Field(
        description=r"""Node field of booleans indicating whether the node belong to an element where the interface is""",
        default_factory=lambda: eval("Field_base()"),
    )
    correspondance_elements: Field_base = Field(
        description=r"""Cell field giving the SALOME cell number""",
        default_factory=lambda: eval("Field_base()"),
    )
    elements_solides: Field_base = Field(
        description=r"""Node field giving the element number containing the solid point""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["ibm_gradient_moyen", "interpolation_ibm_gradient_moyen"],
        "points_solides": [],
        "est_dirichlet": [],
        "correspondance_elements": [],
        "elements_solides": [],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Interpolation_ibm_power_law_tbl_u_star(Interpolation_ibm_mean_gradient):
    r"""
    Immersed Boundary Method (IBM): law u star.
    """

    points_solides: Field_base = Field(
        description=r"""Node field giving the projection of the node on the immersed boundary""",
        default_factory=lambda: eval("Field_base()"),
    )
    est_dirichlet: Field_base = Field(
        description=r"""Node field of booleans indicating whether the node belong to an element where the interface is""",
        default_factory=lambda: eval("Field_base()"),
    )
    correspondance_elements: Field_base = Field(
        description=r"""Cell field giving the SALOME cell number""",
        default_factory=lambda: eval("Field_base()"),
    )
    elements_solides: Field_base = Field(
        description=r"""Node field giving the element number containing the solid point""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["ibm_power_law_tbl_u_star"],
        "points_solides": [],
        "est_dirichlet": [],
        "correspondance_elements": [],
        "elements_solides": [],
        "impr": [],
        "nb_histo_boxes_impr": [],
    }


################################################################


class Partitionneur_deriv(Objet_u):
    r"""
    not_set
    """

    nb_parts: Optional[int] = Field(
        description=r"""The number of non empty parts that must be generated (generally equal to the number of processors in the parallel run).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "nb_parts": []}


################################################################


class Partitionneur_union(Partitionneur_deriv):
    r"""
    Let several local domains be generated from a bigger one using the keyword
    create_domain_from_sub_domain, and let their partitions be generated in the usual way.
    Provided the list of partition files for each small domain, the keyword 'union' will
    partition the global domain in a conform fashion with the smaller domains.
    """

    liste: Bloc_lecture = Field(
        description=r"""List of the partition files with the following syntaxe: {sous_domaine1 decoupage1 ... sous_domaineim decoupageim } where sous_domaine1 ... sous_zomeim are small domains names and decoupage1 ... decoupageim are partition files.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: ["union"], "liste": [], "nb_parts": []}


################################################################


class Bloc_decouper(Objet_lecture):
    r"""
    Auxiliary class to cut a domain.
    """

    partitionneur: Optional[Partitionneur_deriv] = Field(
        description=r"""Defines the partitionning algorithm (the effective C++ object used is \'Partitionneur_ALGORITHM_NAME\').""",
        default=None,
    )
    larg_joint: Optional[int] = Field(
        description=r"""This keyword specifies the thickness of the virtual ghost domaine (data known by one processor though not owned by it). The default value is 1 and is generally correct for all algorithms except the QUICK convection scheme that require a thickness of 2. Since the 1.5.5 version, the VEF discretization imply also a thickness of 2 (except VEF P0). Any non-zero positive value can be used, but the amount of data to store and exchange between processors grows quickly with the thickness.""",
        default=None,
    )
    nom_zones: Optional[str] = Field(
        description=r"""Name of the files containing the different partition of the domain. The files will be :  name_0001.Zones  name_0002.Zones  ...  name_000n.Zones. If this keyword is not specified, the geometry is not written on disk (you might just want to generate a \'ecrire_decoupage\' or \'ecrire_lata\').""",
        default=None,
    )
    ecrire_decoupage: Optional[str] = Field(
        description=r"""After having called the partitionning algorithm, the resulting partition is written on disk in the specified filename. See also partitionneur Fichier_Decoupage. This keyword is useful to change the partition numbers: first, you write the partition into a file with the option ecrire_decoupage. This file contains the domaine number for each element\'s mesh. Then you can easily permute domaine numbers in this file. Then read the new partition to create the .Zones files with the Fichier_Decoupage keyword.""",
        default=None,
    )
    ecrire_lata: Optional[str] = Field(
        description=r"""Save the partition field in a LATA format file for visualization""",
        default=None,
    )
    ecrire_med: Optional[str] = Field(
        description=r"""Save the partition field in a MED format file for visualization""",
        default=None,
    )
    nb_parts_tot: Optional[int] = Field(
        description=r"""Keyword to generates N .Domaine files, instead of the default number M obtained after the partitionning algorithm. N must be greater or equal to M. This option might be used to perform coupled parallel computations. Supplemental empty domaines from M to N-1 are created. This keyword is used when you want to run a parallel calculation on several domains with for example, 2 processors on a first domain and 10 on the second domain because the first domain is very small compare to second one. You will write Nb_parts 2 and Nb_parts_tot 10 for the first domain and Nb_parts 10 for the second domain.""",
        default=None,
    )
    periodique: Optional[List[str]] = Field(
        description=r"""N BOUNDARY_NAME_1 BOUNDARY_NAME_2 ... : N is the number of boundary names given. Periodic boundaries must be declared by this method. The partitionning algorithm will ensure that facing nodes and faces in the periodic boundaries are located on the same processor.""",
        default=None,
    )
    reorder: Optional[int] = Field(
        description=r"""If this option is set to 1 (0 by default), the partition is renumbered in order that the processes which communicate the most are nearer on the network. This may slighlty improves parallel performance.""",
        default=None,
    )
    single_hdf: Optional[bool] = Field(
        description=r"""Optional keyword to enable you to write the partitioned domaines in a single file in hdf5 format.""",
        default=None,
    )
    print_more_infos: Optional[int] = Field(
        description=r"""If this option is set to 1 (0 by default), print infos about number of remote elements (ghosts) and additional infos about the quality of partitionning. Warning, it slows down the cutting operations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "partitionneur": ["partition_tool"],
        "larg_joint": [],
        "nom_zones": ["zones_name"],
        "ecrire_decoupage": [],
        "ecrire_lata": [],
        "ecrire_med": [],
        "nb_parts_tot": [],
        "periodique": [],
        "reorder": [],
        "single_hdf": [],
        "print_more_infos": [],
    }


################################################################


class Partition(Interprete):
    r"""
    Class for parallel calculation to cut a domain for each processor. By default, this
    keyword is commented in the reference test cases.
    """

    domaine: str = Field(description=r"""Name of the domain to be cut.""", default="")
    bloc_decouper: Bloc_decouper = Field(
        description=r"""Description how to cut a domain.""",
        default_factory=lambda: eval("Bloc_decouper()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["partition_64", "decouper"],
        "domaine": [],
        "bloc_decouper": [],
    }


################################################################


class Partitionneur_fichier_med(Partitionneur_deriv):
    r"""
    Partitioning a domain using a MED file containing an integer field providing for each
    element the processor number on which the element should be located.
    """

    file: str = Field(description=r"""file name of the MED file to load""", default="")
    field: Optional[str] = Field(
        description=r"""field name of the integer (or double) field to load""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["fichier_med"],
        "file": [],
        "field": [],
        "nb_parts": [],
    }


################################################################


class Partitionneur_partition(Partitionneur_deriv):
    r"""
    This algorithm re-use the partition of the domain named DOMAINE_NAME. It is useful to
    partition for example a post processing domain. The partition should match with the
    calculation domain.
    """

    domaine: str = Field(description=r"""domain name""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["partition_64", "partition", "decouper"],
        "domaine": [],
        "nb_parts": [],
    }


################################################################


class Partitionneur_sous_domaines(Partitionneur_deriv):
    r"""
    This algorithm will create one part for each specified subdomaine/domain. All elements
    contained in the first subdomaine/domain are put in the first part, all remaining elements
    contained in the second subdomaine/domain in the second part, etc...

    If all elements of the current domain are contained in the specified subdomaines/domain,
    then N parts are created, otherwise, a supplemental part is created with the remaining
    elements.

    If no subdomaine is specified, all subdomaines defined in the domain are used to split
    the mesh.
    """

    sous_zones: Optional[List[str]] = Field(
        description=r"""N SUBZONE_NAME_1 SUBZONE_NAME_2 ...""", default=None
    )
    domaines: Optional[List[str]] = Field(
        description=r"""N DOMAIN_NAME_1 DOMAIN_NAME_2 ...""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["partitionneur_sous_zones", "sous_zones"],
        "sous_zones": [],
        "domaines": [],
        "nb_parts": [],
    }


################################################################


class Partitionneur_tranche(Partitionneur_deriv):
    r"""
    This algorithm will create a geometrical partitionning by slicing the mesh in the two or
    three axis directions, based on the geometric center of each mesh element. nz must be
    given if dimension=3. Each slice contains the same number of elements (slices don\'t have
    the same geometrical width, and for VDF meshes, slice boundaries are generally not flat
    except if the number of mesh elements in each direction is an exact multiple of the number
    of slices). First, nx slices in the X direction are created, then each slice is split in
    ny slices in the Y direction, and finally, each part is split in nz slices in the Z
    direction. The resulting number of parts is nx*ny*nz. If one particular direction has been
    declared periodic, the default slicing (0, 1, 2, ..., n-1)is replaced by (0, 1, 2, ...
    n-1, 0), each of the two \'0\' slices having twice less elements than the other slices.
    """

    tranches: Optional[Annotated[List[int], "size_is_dim"]] = Field(
        description=r"""Partitioned by nx in the X direction, ny in the Y direction, nz in the Z direction. Works only for structured meshes. No warranty for unstructured meshes.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["tranche"], "tranches": [], "nb_parts": []}


################################################################


class Partitionneur_sous_dom(Partitionneur_deriv):
    r"""
    Given a global partition of a global domain, 'sous-domaine' allows to produce a conform
    partition of a sub-domain generated from the bigger one using the keyword
    create_domain_from_sub_domain. The sub-domain will be partitionned in a conform fashion
    with the global domain.
    """

    fichier: str = Field(description=r"""fichier""", default="")
    fichier_ssz: str = Field(description=r"""fichier sous zonne""", default="")
    _synonyms: ClassVar[dict] = {
        None: ["sous_dom"],
        "fichier": [],
        "fichier_ssz": [],
        "nb_parts": [],
    }


################################################################


class Partitionneur_metis(Partitionneur_deriv):
    r"""
    Metis is an external partitionning library. It is a general algorithm that will generate a
    partition of the domain.
    """

    kmetis: Optional[bool] = Field(
        description=r"""The default values are pmetis, default parameters are automatically chosen by Metis. \'kmetis\' is faster than pmetis option but the last option produces better partitioning quality. In both cases, the partitioning quality may be slightly improved by increasing the nb_essais option (by default N=1). It will compute N partitions and will keep the best one (smallest edge cut number). But this option is CPU expensive, taking N=10 will multiply the CPU cost of partitioning by 10.  Experiments show that only marginal improvements can be obtained with non default parameters.""",
        default=None,
    )
    use_weights: Optional[bool] = Field(
        description=r"""If use_weights is specified, weighting of the element-element links in the graph is used to force metis to keep opposite periodic elements on the same processor. This option can slightly improve the partitionning quality but it consumes more memory and takes more time. It is not mandatory since a correction algorithm is always applied afterwards to ensure a correct partitionning for periodic boundaries.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["metis"],
        "kmetis": [],
        "use_weights": [],
        "nb_parts": [],
    }


################################################################


class Partition_multi(Interprete):
    r"""
    allows to partition multiple domains in contact with each other in parallel: necessary for
    resolution monolithique in implicit schemes and for all coupled problems using
    PolyMAC_P0P1NC. By default, this keyword is commented in the reference test cases.
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    domaine1: Literal["domaine"] = Field(description=r"""not set.""", default="domaine")
    dom: str = Field(description=r"""Name of the first domain to be cut.""", default="")
    blocdecoupdom1: Bloc_decouper = Field(
        description=r"""Partition bloc for the first domain.""",
        default_factory=lambda: eval("Bloc_decouper()"),
    )
    domaine2: Literal["domaine"] = Field(description=r"""not set.""", default="domaine")
    dom2: str = Field(
        description=r"""Name of the second domain to be cut.""", default=""
    )
    blocdecoupdom2: Bloc_decouper = Field(
        description=r"""Partition bloc for the second domain.""",
        default_factory=lambda: eval("Bloc_decouper()"),
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["decouper_multi"],
        "aco": [],
        "domaine1": [],
        "dom": [],
        "blocdecoupdom1": [],
        "domaine2": [],
        "dom2": [],
        "blocdecoupdom2": [],
        "acof": [],
    }


################################################################


class Partitionneur_fichier_decoupage(Partitionneur_deriv):
    r"""
    This algorithm reads an array of integer values on the disc, one value for each mesh
    element. Each value is interpreted as the target part number n>=0 for this element. The
    number of parts created is the highest value in the array plus one. Empty parts can be
    created if some values are not present in the array.

    The file format is ASCII, and contains space, tab or carriage-return separated integer
    values. The first value is the number nb_elem of elements in the domain, followed by
    nb_elem integer values (positive or zero).

    This algorithm has been designed to work together with the \'ecrire_decoupage\' option.
    You can generate a partition with any other algorithm, write it to disc, modify it, and
    read it again to generate the .Zone files.

    Contrary to other partitioning algorithms, no correction is applied by default to the
    partition (eg. element 0 on processor 0 and corrections for periodic boundaries). If
    \'corriger_partition\' is specified, these corrections are applied.
    """

    fichier: str = Field(description=r"""File name""", default="")
    corriger_partition: Optional[bool] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["fichier_decoupage"],
        "fichier": [],
        "corriger_partition": [],
        "nb_parts": [],
    }


################################################################


class Point(Points):
    r"""
    Point as class-daughter of Points.
    """

    _synonyms: ClassVar[dict] = {None: [], "points": []}


################################################################


class Segment(Sonde_base):
    r"""
    Keyword to define the number of probe segment points. The file is arranged in columns.
    """

    nbr: int = Field(
        description=r"""Number of probe points of the segment, evenly distributed.""",
        default=0,
    )
    point_deb: Un_point = Field(
        description=r"""First outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    point_fin: Un_point = Field(
        description=r"""Second outer probe segment point.""",
        default_factory=lambda: eval("Un_point()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nbr": [], "point_deb": [], "point_fin": []}


################################################################


class Boundary_field_inward(Front_field_base):
    r"""
    this field is used to define the normal vector field standard at the boundary in VDF or
    VEF discretization.
    """

    normal_value: str = Field(
        description=r"""normal vector value (positive value for a vector oriented outside to inside) which can depend of the time.""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "normal_value": []}


################################################################


class Champ_front_normal_vef(Front_field_base):
    r"""
    Field to define the normal vector field standard at the boundary in VEF discretization.
    """

    mot: Literal["valeur_normale"] = Field(
        description=r"""Name of vector field.""", default="valeur_normale"
    )
    vit_tan: float = Field(
        description=r"""normal vector value (positive value for a vector oriented outside to inside).""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {None: [], "mot": [], "vit_tan": []}


################################################################


class Champ_fonc_tabule(Champ_don_base):
    r"""
    Field that is tabulated as a function of another field.
    """

    pb_field: Bloc_lecture = Field(
        description=r"""block similar to { pb1 field1 } or { pb1 field1 ... pbN fieldN }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    dim: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""Values (the table (the value of the field at any time is calculated by linear interpolation from this table) or the analytical expression (with keyword expression to use an analytical expression)).""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "pb_field": [], "dim": [], "bloc": []}


################################################################


class Champ_fonc_fonction(Champ_fonc_tabule):
    r"""
    Field that is a function of another field.
    """

    dim: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    pb_field: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    bloc: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    problem_name: str = Field(description=r"""Name of problem.""", default="")
    inco: str = Field(
        description=r"""Name of the field (for example: temperature).""", default=""
    )
    expression: List[str] = Field(
        description=r"""Number of field components followed by the analytical expression for each field component.""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "problem_name": [],
        "inco": [],
        "expression": [],
    }


################################################################


class Champ_front_debit_massique(Front_field_base):
    r"""
    This field is used to define a flow rate field using the density
    """

    ch: Front_field_base = Field(
        description=r"""uniform field in space to define the flow rate. It could be, for example, champ_front_uniforme, ch_front_input_uniform or champ_front_fonc_txyz that depends only on time.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Champ_post_de_champs_post(Champ_generique_base):
    r"""
    not_set
    """

    source: Optional[Champ_generique_base] = Field(
        description=r"""the source field.""", default=None
    )
    sources: Optional[Annotated[List[Champ_generique_base], "Listchamp_generique"]] = (
        Field(description=r"""XXX""", default=None)
    )
    nom_source: Optional[str] = Field(
        description=r"""To name a source field with the nom_source keyword""",
        default=None,
    )
    source_reference: Optional[str] = Field(description=r"""not_set""", default=None)
    sources_reference: Optional[Annotated[List[Nom_anonyme], "List_nom_virgule"]] = (
        Field(description=r"""List of name.""", default=None)
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_post_statistiques_base(Champ_post_de_champs_post):
    r"""
    not_set
    """

    t_deb: float = Field(description=r"""Start of integration time""", default=0.0)
    t_fin: float = Field(description=r"""End of integration time""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "t_deb": [],
        "t_fin": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Correlation(Champ_post_statistiques_base):
    r"""
    to calculate the correlation between the two fields.
    """

    _synonyms: ClassVar[dict] = {
        None: ["champ_post_statistiques_correlation"],
        "t_deb": [],
        "t_fin": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_front_bruite(Front_field_base):
    r"""
    Field which is variable in time and space in a random manner.
    """

    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""{ [N val L val ] Moyenne m_1.....[m_i ] Amplitude A_1.....[A_ i ]}: Random nois: If N and L are not defined, the ith component of the field varies randomly around an average value m_i with a maximum amplitude A_i.  White noise: If N and L are defined, these two additional parameters correspond to L, the domain length and N, the number of nodes in the domain. Noise frequency will be between 2*Pi/L and 2*Pi*N/(4*L).  For example, formula for velocity: u=U0(t) v=U1(t)Uj(t)=Mj+2*Aj*bruit_blanc where bruit_blanc (white_noise) is the formula given in the mettre_a_jour (update) method of the Champ_front_bruite (noise_boundary_field) (Refer to the Champ_front_bruite.cpp file).""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nb_comp": [], "bloc": []}


################################################################


class Ecart_type(Champ_post_statistiques_base):
    r"""
    to calculate the standard deviation (statistic rms) of the field nom_champ.
    """

    _synonyms: ClassVar[dict] = {
        None: ["champ_post_statistiques_ecart_type"],
        "t_deb": [],
        "t_fin": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Ch_front_input(Front_field_base):
    r"""
    not_set
    """

    nb_comp: int = Field(description=r"""not_set""", default=0)
    nom: str = Field(description=r"""not_set""", default="")
    initial_value: Optional[List[float]] = Field(
        description=r"""not_set""", default=None
    )
    probleme: str = Field(description=r"""not_set""", default="")
    sous_zone: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "nb_comp": [],
        "nom": [],
        "initial_value": [],
        "probleme": [],
        "sous_zone": [],
    }


################################################################


class Ch_front_input_uniforme(Ch_front_input):
    r"""
    for coupling, you can use ch_front_input_uniforme which is a champ_front_uniforme, which
    use an external value. It must be used with Problem.setInputField.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "nb_comp": [],
        "nom": [],
        "initial_value": [],
        "probleme": [],
        "sous_zone": [],
    }


################################################################


class Champ_front_tabule(Front_field_base):
    r"""
    Constant field on the boundary, tabulated as a function of time.
    """

    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""{nt1 t2 t3 ....tn u1 [v1 w1 ...] u2 [v2 w2 ...] u3 [v3 w3 ...] ... un [vn wn ...] }  Values are entered into a table based on n couples (ti, ui) if nb_comp value is 1. The value of a field at a given time is calculated by linear interpolation from this table.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nb_comp": [], "bloc": []}


################################################################


class Champ_front_tabule_lu(Champ_front_tabule):
    r"""
    Constant field on the boundary, tabulated from a specified column file. Lines starting
    with # are ignored.
    """

    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    column_file: str = Field(description=r"""Name of the column file.""", default="")
    bloc: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    _synonyms: ClassVar[dict] = {None: [], "nb_comp": [], "column_file": []}


################################################################


class Reduction_0d(Champ_post_de_champs_post):
    r"""
    To calculate the min, max, sum, average, weighted sum, weighted average, weighted sum by
    porosity, weighted average by porosity, euclidian norm, normalized euclidian norm, L1
    norm, L2 norm of a field.
    """

    methode: Literal[
        "min",
        "max",
        "moyenne",
        "average",
        "moyenne_ponderee",
        "weighted_average",
        "somme",
        "sum",
        "somme_ponderee",
        "weighted_sum",
        "somme_ponderee_porosite",
        "weighted_sum_porosity",
        "euclidian_norm",
        "normalized_euclidian_norm",
        "l1_norm",
        "l2_norm",
        "valeur_a_gauche",
        "left_value",
    ] = Field(
        description=r"""name of the reduction method:  - min for the minimum value,  - max for the maximum value,  - average (or moyenne) for a mean,  - weighted_average (or moyenne_ponderee) for a mean ponderated by integration volumes, e.g: cell volumes for temperature and pressure in VDF, volumes around faces for velocity and temperature in VEF,  - sum (or somme) for the sum of all the values of the field,  - weighted_sum (or somme_ponderee) for a weighted sum (integral),  - weighted_average_porosity (or moyenne_ponderee_porosite) and weighted_sum_porosity (or somme_ponderee_porosite) for the mean and sum weighted by the volumes of the elements, only for ELEM localisation,  - euclidian_norm for the euclidian norm,  - normalized_euclidian_norm for the euclidian norm normalized,  - L1_norm for norm L1,  - L2_norm for norm L2""",
        default="min",
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_reduction_0d"],
        "methode": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_fonc_fonction_txyz(Champ_fonc_fonction):
    r"""
    this refers to a field that is a function of another field and time and/or space
    coordinates
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "problem_name": [],
        "inco": [],
        "expression": [],
    }


################################################################


class Interpolation(Champ_post_de_champs_post):
    r"""
    To create a field which is an interpolation of the field given by the keyword source.
    """

    localisation: str = Field(
        description=r"""type_loc indicate where is done the interpolation (elem for element or som for node).""",
        default="",
    )
    methode: Optional[str] = Field(
        description=r"""The optional keyword methode is limited to calculer_champ_post for the moment.""",
        default=None,
    )
    domaine: Optional[str] = Field(
        description=r"""the domain name where the interpolation is done (by default, the calculation domain)""",
        default=None,
    )
    optimisation_sous_maillage: Optional[
        Literal[
            "default",
            "yes",
            "no",
        ]
    ] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_interpolation"],
        "localisation": [],
        "methode": [],
        "domaine": [],
        "optimisation_sous_maillage": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_front_debit(Front_field_base):
    r"""
    This field is used to define a flow rate field instead of a velocity field for a Dirichlet
    boundary condition on Navier-Stokes equations.
    """

    ch: Front_field_base = Field(
        description=r"""uniform field in space to define the flow rate. It could be, for example, champ_front_uniforme, ch_front_input_uniform or champ_front_fonc_txyz that depends only on time.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Transformation(Champ_post_de_champs_post):
    r"""
    To create a field with a transformation using source fields and x, y, z, t. If you use in
    your datafile source refChamp { Pb_champ pb pression }, the field pression may be used in
    the expression with the name pression_natif_dom; this latter is the same as pression. If
    you specify nom_source in refChamp bloc, you should use the alias given to pressure field.
    This is avail for all equations unknowns in transformation.
    """

    methode: Literal[
        "produit_scalaire", "norme", "vecteur", "formule", "composante"
    ] = Field(
        description=r"""methode 0 methode norme : will calculate the norm of a vector given by a source field  methode produit_scalaire : will calculate the dot product of two vectors given by two sources fields  methode composante numero integer : will create a field by extracting the integer component of a field given by a source field  methode formule expression 1 : will create a scalar field located to elements using expressions with x,y,z,t parameters and field names given by a source field or several sources fields.  methode vecteur expression N f1(x,y,z,t) fN(x,y,z,t) : will create a vector field located to elements by defining its N components with N expressions with x,y,z,t parameters and field names given by a source field or several sources fields.""",
        default="produit_scalaire",
    )
    unite: Optional[str] = Field(
        description=r"""will specify the field unit""", default=None
    )
    expression: Optional[List[str]] = Field(
        description=r"""expression 1 see methodes formule and vecteur""", default=None
    )
    numero: Optional[int] = Field(
        description=r"""numero 1 see methode composante""", default=None
    )
    localisation: Optional[str] = Field(
        description=r"""localisation 1 type_loc indicate where is done the interpolation (elem for element or som for node). The optional keyword methode is limited to calculer_champ_post for the moment""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_transformation"],
        "methode": [],
        "unite": [],
        "expression": [],
        "numero": [],
        "localisation": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_front_xyz_debit(Front_field_base):
    r"""
    This field is used to define a flow rate field with a velocity profil which will be
    normalized to match the flow rate chosen.
    """

    velocity_profil: Optional[Front_field_base] = Field(
        description=r"""velocity_profil 0 velocity field to define the profil of velocity.""",
        default=None,
    )
    flow_rate: Front_field_base = Field(
        description=r"""flow_rate 1 uniform field in space to define the flow rate. It could be, for example, champ_front_uniforme, ch_front_input_uniform or champ_front_fonc_t""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "velocity_profil": [], "flow_rate": []}


################################################################


class Champ_front_fonc_pois_tube(Front_field_base):
    r"""
    Boundary field champ_front_fonc_pois_tube.
    """

    r_tube: float = Field(description=r"""not_set""", default=0.0)
    umoy: List[float] = Field(description=r"""not_set""", default_factory=list)
    r_loc: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""not_set""", default_factory=list
    )
    r_loc_mult: Annotated[List[int], "size_is_dim"] = Field(
        description=r"""not_set""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "r_tube": [],
        "umoy": [],
        "r_loc": [],
        "r_loc_mult": [],
    }


################################################################


class Champ_front_lu(Front_field_base):
    r"""
    boundary field which is given from data issued from a read file. The format of this file
    has to be the same that the one generated by Ecrire_fichier_xyz_valeur

    Example for K and epsilon quantities to be defined for inlet condition in a boundary
    named \'entree\':

    entree frontiere_ouverte_K_Eps_impose Champ_Front_lu dom 2pb_K_EPS_PERIO_1006.306198.dat
    """

    domaine: str = Field(description=r"""Name of domain""", default="")
    dim: int = Field(description=r"""number of components""", default=0)
    file: str = Field(description=r"""path for the read file""", default="")
    _synonyms: ClassVar[dict] = {None: [], "domaine": ["domain"], "dim": [], "file": []}


################################################################


class Moyenne(Champ_post_statistiques_base):
    r"""
    to calculate the average of the field over time
    """

    moyenne_convergee: Optional[Field_base] = Field(
        description=r"""This option allows to read a converged time averaged field in a .xyz file in order to calculate, when resuming the calculation, the statistics fields (rms, correlation) which depend on this average. In that case, the time averaged field is not updated during the resume of calculation. In this case, the time averaged field must be fully converged to avoid errors when calculating high order statistics.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_statistiques_moyenne"],
        "moyenne_convergee": [],
        "t_deb": [],
        "t_fin": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Refchamp(Champ_generique_base):
    r"""
    Field of prolem
    """

    nom_source: Optional[str] = Field(
        description=r"""The alias name for the field""", default=None
    )
    pb_champ: Deuxmots = Field(
        description=r"""{ Pb_champ nom_pb nom_champ } : nom_pb is the problem name and nom_champ is the selected field name.""",
        default_factory=lambda: eval("Deuxmots()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_refchamp"],
        "nom_source": [],
        "pb_champ": [],
    }


################################################################


class Predefini(Champ_generique_base):
    r"""
    This keyword is used to post process predefined postprocessing fields.
    """

    pb_champ: Deuxmots = Field(
        description=r"""{ Pb_champ nom_pb nom_champ } : nom_pb is the problem name and nom_champ is the selected field name. The available keywords for the field name are: energie_cinetique_totale, energie_cinetique_elem, viscosite_turbulente, viscous_force_x, viscous_force_y, viscous_force_z, pressure_force_x, pressure_force_y, pressure_force_z, total_force_x, total_force_y, total_force_z, viscous_force, pressure_force, total_force""",
        default_factory=lambda: eval("Deuxmots()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "pb_champ": []}


################################################################


class Champ_front_composite(Front_field_base):
    r"""
    Composite front field. Used in multiphase problems to associate data to each phase.
    """

    dim: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""Values Various pieces of the field, defined per phase. Part 1 goes to phase 1, etc...""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "bloc": []}


################################################################


class Champ_front_fonc_txyz(Front_field_base):
    r"""
    Boundary field which is not constant in space and in time.
    """

    val: List[str] = Field(
        description=r"""Values of field components (mathematical expressions).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Champ_front_xyz_tabule(Champ_front_fonc_txyz):
    r"""
    Space dependent field on the boundary, tabulated as a function of time.
    """

    val: List[str] = Field(
        description=r"""Values of field components (mathematical expressions).""",
        default_factory=list,
    )
    bloc: Bloc_lecture = Field(
        description=r"""{nt1 t2 t3 ....tn u1 [v1 w1 ...] u2 [v2 w2 ...] u3 [v3 w3 ...] ... un [vn wn ...] }  Values are entered into a table based on n couples (ti, ui) if nb_comp value is 1. The value of a field at a given time is calculated by linear interpolation from this table.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "val": [], "bloc": []}


################################################################


class Champ_front_fonc_t(Front_field_base):
    r"""
    Boundary field that depends only on time.
    """

    val: List[str] = Field(
        description=r"""Values of field components (mathematical expressions).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Champ_post_operateur_base(Champ_post_de_champs_post):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Divergence(Champ_post_operateur_base):
    r"""
    To calculate divergency of a given field.
    """

    _synonyms: ClassVar[dict] = {
        None: ["champ_post_operateur_divergence"],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_front_musig(Champ_front_composite):
    r"""
    MUSIG front field. Used in multiphase problems to associate data to each phase.
    """

    dim: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    bloc: Bloc_lecture = Field(
        description=r"""Not set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Champ_front_calc(Front_field_base):
    r"""
    This keyword is used on a boundary to get a field from another boundary. The local and
    remote boundaries should have the same mesh. If not, the Champ_front_recyclage keyword
    could be used instead. It is used in the condition block at the limits of equation which
    itself refers to a problem called pb1. We are working under the supposition that pb1 is
    coupled to another problem.
    """

    problem_name: str = Field(
        description=r"""Name of the other problem to which pb1 is coupled.""",
        default="",
    )
    bord: str = Field(
        description=r"""Name of the side which is the boundary between the 2 domains in the domain object description associated with the problem_name object.""",
        default="",
    )
    field_name: str = Field(
        description=r"""Name of the field containing the value that the user wishes to use at the boundary. The field_name object must be recognized by the problem_name object.""",
        default="",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "problem_name": [],
        "bord": [],
        "field_name": [],
    }


################################################################


class Champ_front_fonc_pois_ipsn(Front_field_base):
    r"""
    Boundary field champ_front_fonc_pois_ipsn.
    """

    r_tube: float = Field(description=r"""not_set""", default=0.0)
    umoy: List[float] = Field(description=r"""not_set""", default_factory=list)
    r_loc: Annotated[List[float], "size_is_dim"] = Field(
        description=r"""not_set""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "r_tube": [], "umoy": [], "r_loc": []}


################################################################


class Champ_input_base(Field_base):
    r"""
    not_set
    """

    nb_comp: int = Field(description=r"""not_set""", default=0)
    nom: str = Field(description=r"""not_set""", default="")
    initial_value: Optional[List[float]] = Field(
        description=r"""not_set""", default=None
    )
    probleme: str = Field(description=r"""not_set""", default="")
    sous_zone: Optional[str] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "nb_comp": [],
        "nom": [],
        "initial_value": [],
        "probleme": [],
        "sous_zone": [],
    }


################################################################


class Champ_input_p0(Champ_input_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "nb_comp": [],
        "nom": [],
        "initial_value": [],
        "probleme": [],
        "sous_zone": [],
    }


################################################################


class Gradient(Champ_post_operateur_base):
    r"""
    To calculate gradient of a given field.
    """

    _synonyms: ClassVar[dict] = {
        None: ["champ_post_operateur_gradient"],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Morceau_equation(Champ_post_de_champs_post):
    r"""
    To calculate a field related to a piece of equation. For the moment, the field which can
    be calculated is the stability time step of an operator equation. The problem name and the
    unknown of the equation should be given by Source refChamp { Pb_Champ problem_name
    unknown_field_of_equation }
    """

    type: str = Field(
        description=r"""can only be operateur for equation operators.""", default=""
    )
    numero: Optional[int] = Field(
        description=r"""numero will be 0 (diffusive operator) or 1 (convective operator) or 2 (gradient operator) or 3 (divergence operator).""",
        default=None,
    )
    unite: Optional[str] = Field(
        description=r"""will specify the field unit""", default=None
    )
    option: Literal["stabilite", "flux_bords", "flux_surfacique_bords"] = Field(
        description=r"""option is stability for time steps or flux_bords for boundary fluxes or flux_surfacique_bords for boundary surfacic fluxes""",
        default="stabilite",
    )
    compo: Optional[int] = Field(
        description=r"""compo will specify the number component of the boundary flux (for boundary fluxes, in this case compo permits to specify the number component of the boundary flux choosen).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_morceau_equation"],
        "type": [],
        "numero": [],
        "unite": [],
        "option": [],
        "compo": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_front_fonction(Front_field_base):
    r"""
    boundary field that is function of another field
    """

    dim: int = Field(description=r"""Number of field components.""", default=0)
    inco: str = Field(
        description=r"""Name of the field (for example: temperature).""", default=""
    )
    expression: str = Field(
        description=r"""keyword to use a analytical expression like 10.*EXP(-0.1*val) where val be the keyword for the field.""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "inco": [], "expression": []}


################################################################


class Champ_front_uniforme(Front_field_base):
    r"""
    Boundary field which is constant in space and stationary.
    """

    val: List[float] = Field(
        description=r"""Values of field components.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Champ_input_p0_composite(Champ_input_base):
    r"""
    Field used to define a classical champ input p0 field (for ICoCo), but with a predefined
    field for the initial state.
    """

    initial_field: Optional[Field_base] = Field(
        description=r"""The field used for initialization""", default=None
    )
    input_field: Optional[Champ_input_p0] = Field(
        description=r"""The input field for ICoCo""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "initial_field": [],
        "input_field": [],
        "nb_comp": [],
        "nom": [],
        "initial_value": [],
        "probleme": [],
        "sous_zone": [],
    }


################################################################


class Extraction(Champ_post_de_champs_post):
    r"""
    To create a surface field (values at the boundary) of a volume field
    """

    domaine: str = Field(description=r"""name of the volume field""", default="")
    nom_frontiere: str = Field(
        description=r"""boundary name where the values of the volume field will be picked""",
        default="",
    )
    methode: Optional[Literal["trace", "champ_frontiere"]] = Field(
        description=r"""name of the extraction method (trace by_default or champ_frontiere)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_post_extraction"],
        "domaine": [],
        "nom_frontiere": [],
        "methode": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Field_func_txyz(Champ_don_base):
    r"""
    Field defined by analytical functions. It makes it possible the definition of a field that
    depends on the time and the space.
    """

    dom: str = Field(description=r"""Name of domain of calculation""", default="")
    val: List[str] = Field(
        description=r"""List of functions on (t,x,y,z).""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["champ_fonc_txyz"], "dom": [], "val": []}


################################################################


class Champ_don_lu(Champ_don_base):
    r"""
    Field to read a data field (values located at the center of the cells) in a file.
    """

    dom: str = Field(description=r"""Name of the domain.""", default="")
    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    file: str = Field(
        description=r"""Name of the file.  This file has the following format:  nb_val_lues -> Number of values readen in th file  Xi Yi Zi -> Coordinates readen in the file  Ui Vi Wi -> Value of the field""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "nb_comp": [], "file": []}


################################################################


class Champ_uniforme_morceaux(Champ_don_base):
    r"""
    Field which is partly constant in space and stationary.
    """

    nom_dom: str = Field(
        description=r"""Name of the domain to which the sub-areas belong.""", default=""
    )
    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    data: Bloc_lecture = Field(
        description=r"""{ Defaut val_def sous_zone_1 val_1 ... sous_zone_i val_i } By default, the value val_def is assigned to the field. It takes the sous_zone_i identifier Sous_Zone (sub_area) type object value, val_i. Sous_Zone (sub_area) type objects must have been previously defined if the operator wishes to use a Champ_Uniforme_Morceaux(partly_uniform_field) type object.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "nb_comp": [], "data": []}


################################################################


class Champ_uniforme_morceaux_tabule_temps(Champ_uniforme_morceaux):
    r"""
    this type of field is constant in space on one or several sub_zones and tabulated as a
    function of time.
    """

    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "nb_comp": [], "data": []}


################################################################


class Tayl_green(Champ_don_base):
    r"""
    Class Tayl_green.
    """

    dim: int = Field(description=r"""Dimension.""", default=0)
    _synonyms: ClassVar[dict] = {None: [], "dim": []}


################################################################


class Champ_tabule_temps(Champ_don_base):
    r"""
    Field that is constant in space and tabulated as a function of time.
    """

    dim: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""Values as a table. The value of the field at any time is calculated by linear interpolation from this table.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "bloc": []}


################################################################


class Champ_fonc_fonction_txyz_morceaux(Champ_don_base):
    r"""
    Field defined by analytical functions in each sub-domaine. On each zone, the value is
    defined as a function of x,y,z,t and of scalar value taken from a parameter field. This
    values is associated to the variable 'val' in the expression.
    """

    problem_name: str = Field(description=r"""Name of the problem.""", default="")
    inco: str = Field(
        description=r"""Name of the field (for example: temperature).""", default=""
    )
    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    data: Bloc_lecture = Field(
        description=r"""{ Defaut val_def sous_domaine_1 val_1 ... sous_domaine_i val_i } By default, the value val_def is assigned to the field. It takes the sous_domaine_i identifier Sous_Domaine (sub_area) type object function, val_i. Sous_Domaine (sub_area) type objects must have been previously defined if the operator wishes to use a champ_fonc_fonction_txyz_morceaux type object.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "problem_name": [],
        "inco": [],
        "nb_comp": [],
        "data": [],
    }


################################################################


class Champ_fonc_tabule_morceaux(Champ_don_base):
    r"""
    Field defined by tabulated data in each sub-domaine. It makes possible the definition of a
    field which is a function of other fields.
    """

    domain_name: str = Field(description=r"""Name of the domain.""", default="")
    nb_comp: int = Field(description=r"""Number of field components.""", default=0)
    data: Bloc_lecture = Field(
        description=r"""{ Defaut val_def sous_domaine_1 val_1 ... sous_domaine_i val_i } By default, the value val_def is assigned to the field. It takes the sous_domaine_i identifier Sous_Domaine (sub_area) type object function, val_i. Sous_Domaine (sub_area) type objects must have been previously defined if the operator wishes to use a champ_fonc_tabule_morceaux type object.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["champ_tabule_morceaux"],
        "domain_name": [],
        "nb_comp": [],
        "data": [],
    }


################################################################


class Init_par_partie(Champ_don_base):
    r"""
    ne marche que pour n_comp=1
    """

    n_comp: Literal[1] = Field(description=r"""not_set""", default=1)
    val1: float = Field(description=r"""not_set""", default=0.0)
    val2: float = Field(description=r"""not_set""", default=0.0)
    val3: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "n_comp": [],
        "val1": [],
        "val2": [],
        "val3": [],
    }


################################################################


class Champ_fonc_t(Champ_don_base):
    r"""
    Field that is constant in space and is a function of time.
    """

    val: List[str] = Field(
        description=r"""Values of field components (time dependant functions).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Champ_composite(Champ_don_base):
    r"""
    Composite field. Used in multiphase problems to associate data to each phase.
    """

    dim: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lecture = Field(
        description=r"""Values Various pieces of the field, defined per phase. Part 1 goes to phase 1, etc...""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "bloc": []}


################################################################


class Champ_musig(Champ_composite):
    r"""
    MUSIG field. Used in multiphase problems to associate data to each phase.
    """

    dim: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    bloc: Bloc_lecture = Field(
        description=r"""Not set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Uniform_field(Champ_don_base):
    r"""
    Field that is constant in space and stationary.
    """

    val: List[float] = Field(
        description=r"""Values of field components.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["champ_uniforme"], "val": []}


################################################################


class Champ_fonc_tabule_morceaux_interp(Champ_fonc_tabule_morceaux):
    r"""
    Field defined by tabulated data in each sub-domaine. It makes possible the definition of a
    field which is a function of other fields. Here we use MEDCoupling to interpolate fields
    between the two domains.
    """

    domain_name: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    problem_name: str = Field(description=r"""Name of the problem.""", default="")
    _synonyms: ClassVar[dict] = {
        None: [],
        "problem_name": [],
        "nb_comp": [],
        "data": [],
    }


################################################################


class Valeur_totale_sur_volume(Champ_uniforme_morceaux):
    r"""
    Similar as Champ_Uniforme_Morceaux with the same syntax. Used for source terms when we
    want to specify a source term with a value given for the volume (eg: heat in Watts) and
    not a value per volume unit (eg: heat in Watts/m3).
    """

    _synonyms: ClassVar[dict] = {None: [], "nom_dom": [], "nb_comp": [], "data": []}


################################################################


class Bloc_lec_champ_init_canal_sinal(Objet_lecture):
    r"""
    Parameters for the class champ_init_canal_sinal.

    in 2D:

    U=ucent*y(2h-y)/h/h

    V=ampli_bruit*rand+ampli_sin*sin(omega*x)

    rand: unpredictable value between -1 and 1.

    in 3D:

    U=ucent*y(2h-y)/h/h

    V=ampli_bruit*rand1+ampli_sin*sin(omega*x)

    W=ampli_bruit*rand2

    rand1 and rand2: unpredictables values between -1 and 1.
    """

    ucent: float = Field(
        description=r"""Velocity value at the center of the channel.""", default=0.0
    )
    h: float = Field(description=r"""Half hength of the channel.""", default=0.0)
    ampli_bruit: float = Field(
        description=r"""Amplitude for the disturbance.""", default=0.0
    )
    ampli_sin: Optional[float] = Field(
        description=r"""Amplitude for the sinusoidal disturbance (by default equals to ucent/10).""",
        default=None,
    )
    omega: float = Field(
        description=r"""Value of pulsation for the of the sinusoidal disturbance.""",
        default=0.0,
    )
    dir_flow: Optional[Literal[0, 1, 2]] = Field(
        description=r"""Flow direction for the initialization of the flow in a channel.  - if dir_flow=0, the flow direction is X  - if dir_flow=1, the flow direction is Y  - if dir_flow=2, the flow direction is Z  Default value for dir_flow is 0""",
        default=None,
    )
    dir_wall: Optional[Literal[0, 1, 2]] = Field(
        description=r"""Wall direction for the initialization of the flow in a channel.  - if dir_wall=0, the normal to the wall is in X direction  - if dir_wall=1, the normal to the wall is in Y direction  - if dir_wall=2, the normal to the wall is in Z direction  Default value for dir_flow is 1""",
        default=None,
    )
    min_dir_flow: Optional[float] = Field(
        description=r"""Value of the minimum coordinate in the flow direction for the initialization of the flow in a channel. Default value for dir_flow is 0.""",
        default=None,
    )
    min_dir_wall: Optional[float] = Field(
        description=r"""Value of the minimum coordinate in the wall direction for the initialization of the flow in a channel. Default value for dir_flow is 0.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "ucent": [],
        "h": [],
        "ampli_bruit": [],
        "ampli_sin": [],
        "omega": [],
        "dir_flow": [],
        "dir_wall": [],
        "min_dir_flow": [],
        "min_dir_wall": [],
    }


################################################################


class Champ_init_canal_sinal(Champ_don_base):
    r"""
    For a parabolic profile on U velocity with an unpredictable disturbance on V and W and a
    sinusoidal disturbance on V velocity.
    """

    dim: int = Field(description=r"""Number of field components.""", default=0)
    bloc: Bloc_lec_champ_init_canal_sinal = Field(
        description=r"""Parameters for the class champ_init_canal_sinal.""",
        default_factory=lambda: eval("Bloc_lec_champ_init_canal_sinal()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dim": [], "bloc": []}


################################################################


class Field_func_xyz(Champ_don_base):
    r"""
    Field defined by analytical functions. It makes it possible the definition of a field that
    depends on (x,y,z).
    """

    dom: str = Field(description=r"""Name of domain of calculation.""", default="")
    val: List[str] = Field(
        description=r"""List of functions on (x,y,z).""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: ["champ_fonc_xyz"], "dom": [], "val": []}


################################################################


class Moyenne_volumique(Interprete):
    r"""
    This keyword should be used after Resoudre keyword. It computes the convolution product of
    one or more fields with a given filtering function.
    """

    nom_pb: str = Field(
        description=r"""name of the problem where the source fields will be searched.""",
        default="",
    )
    nom_domaine: str = Field(
        description=r"""name of the destination domain (for example, it can be a coarser mesh, but for optimal performance in parallel, the domain should be split with the same algorithm as the computation mesh, eg, same tranche parameters for example)""",
        default="",
    )
    noms_champs: List[str] = Field(
        description=r"""name of the source fields (these fields must be accessible from the postraitement) N source_field1 source_field2 ... source_fieldN""",
        default_factory=list,
    )
    format_post: Optional[str] = Field(
        description=r"""gives the fileformat for the result (by default : lata)""",
        default=None,
    )
    nom_fichier_post: Optional[str] = Field(
        description=r"""indicates the filename where the result is written""",
        default=None,
    )
    fonction_filtre: Bloc_lecture = Field(
        description=r"""to specify the given filter  Fonction_filtre { type filter_type demie-largeur l [ omega w ]  [ expression string ] }   type filter_type : This parameter specifies the filtering function. Valid filter_type are: Boite is a box filter, $f(x,y,z)=(abs(x)<l)*(abs(y) <l)*(abs(z) <l) / (8 l^3)$ Chapeau is a hat filter (product of hat filters in each direction) centered on the origin, the half-width of the filter being l and its integral being 1. Quadra is a 2nd order filter. Gaussienne is a normalized gaussian filter of standard deviation sigma in each direction (all field elements outside a cubic box defined by clipping_half_width are ignored, hence, taking clipping_half_width=2.5*sigma yields an integral of 0.99 for a uniform unity field). Parser allows a user defined function of the x,y,z variables. All elements outside a cubic box defined by clipping_half_width are ignored. The parser is much slower than the equivalent c++ coded function...  demie-largeur l : This parameter specifies the half width of the filter [ omega w ] : This parameter must be given for the gaussienne filter. It defines the standard deviation of the gaussian filter. [ expression string] : This parameter must be given for the parser filter type. This expression will be interpreted by the math parser with the predefined variables x, y and z.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    localisation: Optional[Literal["elem", "som"]] = Field(
        description=r"""indicates where the convolution product should be computed: either on the elements or on the nodes of the destination domain.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nom_pb": [],
        "nom_domaine": [],
        "noms_champs": [],
        "format_post": [],
        "nom_fichier_post": [],
        "fonction_filtre": [],
        "localisation": [],
    }


################################################################


class Solveur_petsc_deriv(Objet_u):
    r"""
    Additional information is available in the PETSC documentation:
    https://petsc.org/release/manual/
    """

    seuil: Optional[float] = Field(
        description=r"""corresponds to the iterative solver convergence value. The iterative solver converges when the Euclidean residue standard ||Ax-B|| is less than seuil.""",
        default=None,
    )
    quiet: Optional[bool] = Field(
        description=r"""is a keyword which is used to not displaying any outputs of the solver.""",
        default=None,
    )
    impr: Optional[bool] = Field(
        description=r"""used to request display of the Euclidean residue standard each time this iterates through the conjugated gradient (display to the standard outlet).""",
        default=None,
    )
    rtol: Optional[float] = Field(description=r"""not_set""", default=None)
    atol: Optional[float] = Field(description=r"""not_set""", default=None)
    save_matrix_mtx_format: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Petsc(Solveur_sys_base):
    r"""
    Solver via Petsc API
    """

    solveur: Solveur_petsc_deriv = Field(
        description=r"""solver type and options""",
        default_factory=lambda: eval("Solveur_petsc_deriv()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": []}


################################################################


class Solveur_petsc_lu(Solveur_petsc_deriv):
    r"""
    Several solvers through PETSc API are available.

    TIPS:



    A) Solver for symmetric linear systems (e.g: Pressure system from Navier-Stokes
    equations):

    -The CHOLESKY parallel solver is from MUMPS library. It offers better performance than
    all others solvers if you have enough RAM for your calculation. A parallel calculation on
    a cluster with 4GBytes on each processor, 40000 cells/processor seems the upper limit.
    Seems to be very slow to initialize above 500 cpus/cores.

    -When running a parallel calculation with a high number of cpus/cores (typically more
    than 500) where preconditioner scalabilty is the key for CPU performance, consider
    BICGSTAB with BLOCK_JACOBI_ICC(1) as preconditioner or if not converges, GCP with
    BLOCK_JACOBI_ICC(1) as preconditioner.

    -For other situations, the first choice should be GCP/SSOR. In order to fine tune the
    solver choice, each one of the previous list should be considered. Indeed, the CPU speed
    of a solver depends of a lot of parameters. You may give a try to the OPTIMAL solver to
    help you to find the fastest solver on your study.


    B) Solver for non symmetric linear systems (e.g.: Implicit schemes):

    The BICGSTAB/DIAG solver seems to offer the best performances.
    """

    _synonyms: ClassVar[dict] = {
        None: ["lu"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cholesky_superlu(Solveur_petsc_deriv):
    r"""
    Parallelized Cholesky from SUPERLU_DIST library (less CPU and RAM, efficient than the
    previous one)
    """

    _synonyms: ClassVar[dict] = {
        None: ["cholesky_superlu"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cholesky_pastix(Solveur_petsc_deriv):
    r"""
    Parallelized Cholesky from PASTIX library.
    """

    _synonyms: ClassVar[dict] = {
        None: ["cholesky_pastix"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cholesky_umfpack(Solveur_petsc_deriv):
    r"""
    Sequential Cholesky from UMFPACK library (seems fast).
    """

    _synonyms: ClassVar[dict] = {
        None: ["cholesky_umfpack"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cholesky_out_of_core(Solveur_petsc_deriv):
    r"""
    Same as the previous one but with a written LU decomposition of disk (save RAM memory but
    add an extra CPU cost during Ax=B solve).
    """

    _synonyms: ClassVar[dict] = {
        None: ["cholesky_out_of_core"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_option_cli(Bloc_lecture):
    r"""
    solver
    """

    _synonyms: ClassVar[dict] = {None: ["nul"], "bloc_lecture": []}


################################################################


class Solveur_petsc_cholesky(Solveur_petsc_deriv):
    r"""
    Parallelized version of Cholesky from MUMPS library. This solver accepts an option to
    select a different ordering than the automatic selected one by MUMPS (and printed by using
    the impr option). The possible choices are Metis, Scotch, PT-Scotch or Parmetis. The two
    last options can only be used during a parallel calculation, whereas the two first are
    available for sequential or parallel calculations. It seems that the CPU cost of A=LU
    factorization but also of the backward/forward elimination steps may sometimes be reduced
    by selecting a different ordering (Scotch seems often the best for b/f elimination) than
    the default one.

    Notice that this solver requires a huge amont of memory compared to iterative methods. To
    know how much RAM you will need by core, then use the impr option to have detailled
    informations during the analysis phase and before the factorisation phase (in the
    following output, you will learn that the largest memory is taken by the zeroth CPU with
    108MB):

    Rank of proc needing largest memory in IC facto : 0

    Estimated corresponding MBYTES for IC facto : 108

    Thanks to the following graph, you read that in order to solve for instance a flow on a
    mesh with 2.6e6 cells, you will need to run a parallel calculation on 32 CPUs if you have
    cluster nodes with only 4GB/core (6.2GB*0.42~2.6GB) :

    \includepng{{petscgraph.jpeg}}{{10}}
    """

    save_matrice: Optional[bool] = Field(description=r"""not_set""", default=None)
    save_matrix_petsc_format: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    reduce_ram: Optional[bool] = Field(description=r"""not_set""", default=None)
    cli_quiet: Optional[Solveur_petsc_option_cli] = Field(
        description=r"""not_set""", default=None
    )
    cli: Optional[Solveur_petsc_option_cli] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["cholesky"],
        "save_matrice": ["save_matrix"],
        "save_matrix_petsc_format": [],
        "reduce_ram": [],
        "cli_quiet": [],
        "cli": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cholesky_mumps_blr(Solveur_petsc_deriv):
    r"""
    BLR for (Block Low-Rank)
    """

    reduce_ram: Optional[bool] = Field(description=r"""not_set""", default=None)
    dropping_parameter: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    cli: Optional[Solveur_petsc_option_cli] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["cholesky_mumps_blr"],
        "reduce_ram": [],
        "dropping_parameter": [],
        "cli": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_cli(Solveur_petsc_deriv):
    r"""
    Command Line Interface. Should be used only by advanced users, to access the whole
    solver/preconditioners from the PETSC API. To find all the available options, run your
    calculation with the -ksp_view -help options:

    trust datafile [N] --ksp_view --help

    -pc_type Preconditioner:(one of) none jacobi pbjacobi bjacobi sor lu shell mg eisenstat
    ilu icc cholesky asm ksp composite redundant nn mat fieldsplit galerkin openmp spai hypre
    tfs (PCSetType)

    HYPRE preconditioner options:

    -pc_hypre_type pilut (choose one of) pilut parasails boomeramg

    HYPRE ParaSails Options

    -pc_hypre_parasails_nlevels 1: Number of number of levels (None)

    -pc_hypre_parasails_thresh 0.1: Threshold (None)

    -pc_hypre_parasails_filter 0.1: filter (None)

    -pc_hypre_parasails_loadbal 0: Load balance (None)

    -pc_hypre_parasails_logging: FALSE Print info to screen (None)

    -pc_hypre_parasails_reuse: FALSE Reuse nonzero pattern in preconditioner (None)

    -pc_hypre_parasails_sym nonsymmetric (choose one of) nonsymmetric SPD nonsymmetric,SPD


    Krylov Method (KSP) Options

    -ksp_type Krylov method:(one of) cg cgne stcg gltr richardson chebychev gmres tcqmr bcgs
    bcgsl cgs tfqmr cr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd (KSPSetType)

    -ksp_max_it 10000: Maximum number of iterations (KSPSetTolerances)

    -ksp_rtol 0: Relative decrease in residual norm (KSPSetTolerances)

    -ksp_atol 1e-12: Absolute value of residual norm (KSPSetTolerances)

    -ksp_divtol 10000: Residual norm increase cause divergence (KSPSetTolerances)

    -ksp_converged_use_initial_residual_norm: Use initial residual residual norm for
    computing relative convergence

    -ksp_monitor_singular_value stdout: Monitor singular values (KSPMonitorSet)

    -ksp_monitor_short stdout: Monitor preconditioned residual norm with fewer digits
    (KSPMonitorSet)

    -ksp_monitor_draw: Monitor graphically preconditioned residual norm (KSPMonitorSet)

    -ksp_monitor_draw_true_residual: Monitor graphically true residual norm (KSPMonitorSet)


    Example to use the multigrid method as a solver, not only as a preconditioner:

    Solveur_pression Petsc CLI {-ksp_type richardson -pc_type hypre -pc_hypre_type boomeramg
    -ksp_atol 1.e-7 }
    """

    seuil: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    quiet: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    impr: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    rtol: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    atol: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    save_matrix_mtx_format: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    cli_bloc: Bloc_lecture = Field(
        description=r"""bloc""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: ["cli"], "cli_bloc": []}


################################################################


class Solveur_petsc_cli_quiet(Solveur_petsc_deriv):
    r"""
    solver
    """

    seuil: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    quiet: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    impr: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    rtol: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    atol: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    save_matrix_mtx_format: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    cli_quiet_bloc: Bloc_lecture = Field(
        description=r"""bloc""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: ["cli_quiet"], "cli_quiet_bloc": []}


################################################################


class Preconditionneur_petsc_deriv(Objet_u):
    r"""
    Preconditioners available with petsc solvers
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Solveur_petsc_ibicgstab(Solveur_petsc_deriv):
    r"""
    Improved version of previous one for massive parallel computations (only a single global
    reduction operation instead of the usual 3 or 4).
    """

    precond: Optional[Preconditionneur_petsc_deriv] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["ibicgstab"],
        "precond": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_bicgstab(Solveur_petsc_deriv):
    r"""
    Stabilized Bi-Conjugate Gradient
    """

    precond: Optional[Preconditionneur_petsc_deriv] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["bicgstab"],
        "precond": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_gmres(Solveur_petsc_deriv):
    r"""
    Generalized Minimal Residual
    """

    precond: Optional[Preconditionneur_petsc_deriv] = Field(
        description=r"""not_set""", default=None
    )
    reuse_preconditioner_nb_it_max: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    save_matrix_petsc_format: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    nb_it_max: Optional[int] = Field(
        description=r"""In order to specify a given number of iterations instead of a condition on the residue with the keyword seuil. May be useful when defining a PETSc solver for the implicit time scheme where convergence is very fast: 5 or less iterations seems enough.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["gmres"],
        "precond": [],
        "reuse_preconditioner_nb_it_max": [],
        "save_matrix_petsc_format": [],
        "nb_it_max": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_gcp(Solveur_petsc_deriv):
    r"""
    Preconditioned Conjugate Gradient
    """

    precond: Optional[Preconditionneur_petsc_deriv] = Field(
        description=r"""preconditioner""", default=None
    )
    precond_nul: Optional[bool] = Field(
        description=r"""No preconditioner used, equivalent to precond null { }""",
        default=None,
    )
    rtol: Optional[float] = Field(description=r"""not_set""", default=None)
    reuse_preconditioner_nb_it_max: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    cli: Optional[Solveur_petsc_option_cli] = Field(
        description=r"""not_set""", default=None
    )
    reorder_matrix: Optional[int] = Field(description=r"""not_set""", default=None)
    read_matrix: Optional[bool] = Field(
        description=r"""save_matrix|read_matrix are the keywords to save|read into a file the constant matrix A of the linear system Ax=B solved (eg: matrix from the pressure linear system for an incompressible flow). It is useful when you want to minimize the MPI communications on massive parallel calculation. Indeed, in VEF discretization, the overlapping width (generaly 2, specified with the largeur_joint option in the partition keyword partition) can be reduced to 1, once the matrix has been properly assembled and saved. The cost of the MPI communications in TRUST itself (not in PETSc) will be reduced with length messages divided by 2. So the strategy is:  I) Partition your VEF mesh with a largeur_joint value of 2  II) Run your parallel calculation on 0 time step, to build and save the matrix with the save_matrix option. A file named Matrix_NBROWS_rows_NCPUS_cpus.petsc will be saved to the disk (where NBROWS is the number of rows of the matrix and NCPUS the number of CPUs used).  III) Partition your VEF mesh with a largeur_joint value of 1  IV) Run your parallel calculation completly now and substitute the save_matrix option by the read_matrix option. Some interesting gains have been noticed when the cost of linear system solve with PETSc is small compared to all the other operations.""",
        default=None,
    )
    save_matrice: Optional[bool] = Field(
        description=r"""see read_matrix""", default=None
    )
    petsc_decide: Optional[int] = Field(description=r"""not_set""", default=None)
    pcshell: Optional[str] = Field(description=r"""not_set""", default=None)
    aij: Optional[bool] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["gcp"],
        "precond": [],
        "precond_nul": [],
        "rtol": [],
        "reuse_preconditioner_nb_it_max": [],
        "cli": [],
        "reorder_matrix": [],
        "read_matrix": [],
        "save_matrice": ["save_matrix"],
        "petsc_decide": [],
        "pcshell": [],
        "aij": [],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Solveur_petsc_pipecg(Solveur_petsc_deriv):
    r"""
    Pipelined Conjugate Gradient (possible reduced CPU cost during massive parallel
    calculation due to a single non-blocking reduction per iteration, if TRUST is built with a
    MPI-3 implementation)... no example in TRUST
    """

    _synonyms: ClassVar[dict] = {
        None: ["pipecg"],
        "seuil": [],
        "quiet": [],
        "impr": [],
        "rtol": [],
        "atol": [],
        "save_matrix_mtx_format": [],
    }


################################################################


class Preconditionneur_petsc_diag(Preconditionneur_petsc_deriv):
    r"""
    Diagonal (Jacobi) preconditioner.
    """

    _synonyms: ClassVar[dict] = {None: ["diag"]}


################################################################


class Preconditionneur_petsc_c_amg(Preconditionneur_petsc_deriv):
    r"""
    preconditionner
    """

    _synonyms: ClassVar[dict] = {None: ["c-amg"]}


################################################################


class Preconditionneur_petsc_sa_amg(Preconditionneur_petsc_deriv):
    r"""
    preconditionner
    """

    _synonyms: ClassVar[dict] = {None: ["sa-amg"]}


################################################################


class Preconditionneur_petsc_block_jacobi_icc(Preconditionneur_petsc_deriv):
    r"""
    Incomplete Cholesky factorization for symmetric matrix with the PETSc implementation.
    """

    level: Optional[int] = Field(
        description=r"""factorization level (default value, 1). In parallel, the factorization is done by block (one per processor by default).""",
        default=None,
    )
    ordering: Optional[Literal["natural", "rcm"]] = Field(
        description=r"""The ordering of the local matrix is natural by default, but rcm ordering, which reduces the bandwith of the local matrix, may interestingly improves the quality of the decomposition and reduces the number of iterations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["block_jacobi_icc"],
        "level": [],
        "ordering": [],
    }


################################################################


class Preconditionneur_petsc_boomeramg(Preconditionneur_petsc_deriv):
    r"""
    Multigrid preconditioner (no option is available yet, look at CLI command and Petsc
    documentation to try other options).
    """

    _synonyms: ClassVar[dict] = {None: ["boomeramg"]}


################################################################


class Preconditionneur_petsc_null(Preconditionneur_petsc_deriv):
    r"""
    No preconditioner used
    """

    _synonyms: ClassVar[dict] = {None: ["null"]}


################################################################


class Preconditionneur_petsc_lu(Preconditionneur_petsc_deriv):
    r"""
    preconditionner
    """

    _synonyms: ClassVar[dict] = {None: ["lu"]}


################################################################


class Preconditionneur_petsc_jacobi(Preconditionneur_petsc_deriv):
    r"""
    preconditionner
    """

    _synonyms: ClassVar[dict] = {None: ["jacobi"]}


################################################################


class Preconditionneur_petsc_eisentat(Preconditionneur_petsc_deriv):
    r"""
    SSOR version with Eisenstat trick which reduces the number of computations and thus CPU
    cost...
    """

    omega: Optional[float] = Field(description=r"""relaxation factor""", default=None)
    _synonyms: ClassVar[dict] = {None: ["eisentat"], "omega": []}


################################################################


class Preconditionneur_petsc_ssor(Preconditionneur_petsc_deriv):
    r"""
    Symmetric Successive Over Relaxation algorithm.
    """

    omega: Optional[float] = Field(
        description=r"""relaxation factor (default value, 1.5)""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["ssor"], "omega": []}


################################################################


class Preconditionneur_petsc_block_jacobi_ilu(Preconditionneur_petsc_deriv):
    r"""
    preconditionner
    """

    level: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["block_jacobi_ilu"], "level": []}


################################################################


class Preconditionneur_petsc_spai(Preconditionneur_petsc_deriv):
    r"""
    Spai Approximate Inverse algorithm from Parasails Hypre library.
    """

    level: Optional[int] = Field(description=r"""first parameter""", default=None)
    epsilon: Optional[float] = Field(description=r"""second parameter""", default=None)
    _synonyms: ClassVar[dict] = {None: ["spai"], "level": [], "epsilon": []}


################################################################


class Preconditionneur_petsc_pilut(Preconditionneur_petsc_deriv):
    r"""
    Dual Threashold Incomplete LU factorization.
    """

    level: Optional[int] = Field(description=r"""factorization level""", default=None)
    epsilon: Optional[float] = Field(description=r"""drop tolerance""", default=None)
    _synonyms: ClassVar[dict] = {None: ["pilut"], "level": [], "epsilon": []}


################################################################


class Precond_base(Objet_u):
    r"""
    Basic class for preconditioning.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Solv_gcp(Solveur_sys_base):
    r"""
    Preconditioned conjugated gradient.
    """

    seuil: float = Field(
        description=r"""Value of the final residue. The gradient ceases iteration when the Euclidean residue standard ||Ax-B|| is less than this value.""",
        default=0.0,
    )
    nb_it_max: Optional[int] = Field(
        description=r"""Keyword to set the maximum iterations number for the Gcp.""",
        default=None,
    )
    impr: Optional[bool] = Field(
        description=r"""Keyword which is used to request display of the Euclidean residue standard each time this iterates through the conjugated gradient (display to the standard outlet).""",
        default=None,
    )
    quiet: Optional[bool] = Field(
        description=r"""To not displaying any outputs of the solver.""", default=None
    )
    save_matrice: Optional[bool] = Field(
        description=r"""to save the matrix in a file.""", default=None
    )
    precond: Optional[Precond_base] = Field(
        description=r"""Keyword to define system preconditioning in order to accelerate resolution by the conjugated gradient. Many parallel preconditioning methods are not equivalent to their sequential counterpart, and you should therefore expect differences, especially when you select a high value of the final residue (seuil). The result depends on the number of processors and on the mesh splitting. It is sometimes useful to run the solver with no preconditioning at all. In particular:  - when the solver does not converge during initial projection,  - when comparing sequential and parallel computations.  With no preconditioning, except in some particular cases (no open boundary), the sequential and the parallel computations should provide exactly the same results within fpu accuracy. If not, there might be a coding error or the system of equations is singular.""",
        default=None,
    )
    precond_nul: Optional[bool] = Field(
        description=r"""Keyword to not use a preconditioning method.""", default=None
    )
    optimized: Optional[bool] = Field(
        description=r"""This keyword triggers a memory and network optimized algorithms useful for strong scaling (when computing less than 100 000 elements per processor). The matrix and the vectors are duplicated, common items removed and only virtual items really used in the matrix are exchanged. Warning: this is experimental and known to fail in some VEF computations (L2 projection step will not converge). Works well in VDF.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["gcp"],
        "seuil": [],
        "nb_it_max": [],
        "impr": [],
        "quiet": [],
        "save_matrice": ["save_matrix"],
        "precond": [],
        "precond_nul": [],
        "optimized": [],
    }


################################################################


class Gcp_ns(Solv_gcp):
    r"""
    not_set
    """

    solveur0: Solveur_sys_base = Field(
        description=r"""Solver type.""",
        default_factory=lambda: eval("Solveur_sys_base()"),
    )
    solveur1: Solveur_sys_base = Field(
        description=r"""Solver type.""",
        default_factory=lambda: eval("Solveur_sys_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "solveur0": [],
        "solveur1": [],
        "seuil": [],
        "nb_it_max": [],
        "impr": [],
        "quiet": [],
        "save_matrice": ["save_matrix"],
        "precond": [],
        "precond_nul": [],
        "optimized": [],
    }


################################################################


class Ssor_bloc(Precond_base):
    r"""
    not_set
    """

    precond0: Optional[Precond_base] = Field(description=r"""not_set""", default=None)
    precond1: Optional[Precond_base] = Field(description=r"""not_set""", default=None)
    preconda: Optional[Precond_base] = Field(description=r"""not_set""", default=None)
    alpha_0: Optional[float] = Field(description=r"""not_set""", default=None)
    alpha_1: Optional[float] = Field(description=r"""not_set""", default=None)
    alpha_a: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "precond0": [],
        "precond1": [],
        "preconda": [],
        "alpha_0": [],
        "alpha_1": [],
        "alpha_a": [],
    }


################################################################


class Precondsolv(Precond_base):
    r"""
    not_set
    """

    solveur: Solveur_sys_base = Field(
        description=r"""Solver type.""",
        default_factory=lambda: eval("Solveur_sys_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": []}


################################################################


class Ssor(Precond_base):
    r"""
    Symmetric successive over-relaxation algorithm.
    """

    omega: Optional[float] = Field(
        description=r"""Over-relaxation facteur (between 1 and 2, default value 1.6).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "omega": []}


################################################################


class Rocalution(Petsc):
    r"""
    Solver via rocALUTION API
    """

    solveur: str = Field(description=r"""not_set""", default="")
    option_solveur: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": [], "option_solveur": []}


################################################################


class Ilu(Precond_base):
    r"""
    This preconditionner can be only used with the generic GEN solver.
    """

    type: Optional[int] = Field(
        description=r"""values can be 0|1|2|3 for null|left|right|left-and-right preconditionning (default value = 2)""",
        default=None,
    )
    filling: Optional[int] = Field(description=r"""default value = 1.""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "type": [], "filling": []}


################################################################


class Amgx(Petsc):
    r"""
    Solver via AmgX API
    """

    solveur: str = Field(description=r"""not_set""", default="")
    option_solveur: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": [], "option_solveur": []}


################################################################


class Amg(Solveur_sys_base):
    r"""
    Wrapper for AMG preconditioner-based solver which switch for the best one on CPU/GPU
    Nvidia/GPU AMD
    """

    solveur: str = Field(description=r"""not_set""", default="")
    option_solveur: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "solveur": [], "option_solveur": []}


################################################################


class Petsc_gpu(Petsc):
    r"""
    GPU solver via Petsc API
    """

    solveur: str = Field(description=r"""not_set""", default="")
    option_solveur: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    atol: Optional[float] = Field(
        description=r"""Absolute threshold for convergence (same as seuil option)""",
        default=None,
    )
    rtol: Optional[float] = Field(
        description=r"""Relative threshold for convergence""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "solveur": [],
        "option_solveur": [],
        "atol": [],
        "rtol": [],
    }


################################################################


class Cholesky(Solveur_sys_base):
    r"""
    Cholesky direct method.
    """

    impr: Optional[bool] = Field(
        description=r"""Keyword which may be used to print the resolution time.""",
        default=None,
    )
    quiet: Optional[bool] = Field(
        description=r"""To disable printing of information""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "impr": [], "quiet": []}


################################################################


class Gen(Solveur_sys_base):
    r"""
    not_set
    """

    solv_elem: str = Field(
        description=r"""To specify a solver among gmres or bicgstab.""", default=""
    )
    precond: Precond_base = Field(
        description=r"""The only preconditionner that we can specify is ilu.""",
        default_factory=lambda: eval("Precond_base()"),
    )
    seuil: Optional[float] = Field(
        description=r"""Value of the final residue. The solver ceases iterations when the Euclidean residue standard ||Ax-B|| is less than this value. default value 1e-12.""",
        default=None,
    )
    impr: Optional[bool] = Field(
        description=r"""Keyword which is used to request display of the Euclidean residue standard each time this iterates through the conjugated gradient (display to the standard outlet).""",
        default=None,
    )
    save_matrice: Optional[bool] = Field(
        description=r"""To save the matrix in a file.""", default=None
    )
    quiet: Optional[bool] = Field(
        description=r"""To not displaying any outputs of the solver.""", default=None
    )
    nb_it_max: Optional[int] = Field(
        description=r"""Keyword to set the maximum iterations number for the GEN solver.""",
        default=None,
    )
    force: Optional[bool] = Field(
        description=r"""Keyword to set ipar[5]=-1 in the GEN solver. This is helpful if you notice that the solver does not perform more than 100 iterations. If this keyword is specified in the datafile, you should provide nb_it_max.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "solv_elem": [],
        "precond": [],
        "seuil": [],
        "impr": [],
        "save_matrice": ["save_matrix"],
        "quiet": [],
        "nb_it_max": [],
        "force": [],
    }


################################################################


class Test_solveur(Interprete):
    r"""
    To test several solvers
    """

    fichier_secmem: Optional[str] = Field(
        description=r"""Filename containing the second member B""", default=None
    )
    fichier_matrice: Optional[str] = Field(
        description=r"""Filename containing the matrix A""", default=None
    )
    fichier_solution: Optional[str] = Field(
        description=r"""Filename containing the solution x""", default=None
    )
    nb_test: Optional[int] = Field(
        description=r"""Number of tests to measure the time resolution (one preconditionnement)""",
        default=None,
    )
    impr: Optional[bool] = Field(
        description=r"""To print the convergence solver""", default=None
    )
    solveur: Optional[Solveur_sys_base] = Field(
        description=r"""To specify a solver""", default=None
    )
    fichier_solveur: Optional[str] = Field(
        description=r"""To specify a file containing a list of solvers""", default=None
    )
    genere_fichier_solveur: Optional[float] = Field(
        description=r"""To create a file of the solver with a threshold convergence""",
        default=None,
    )
    seuil_verification: Optional[float] = Field(
        description=r"""Check if the solution satisfy ||Ax-B||<precision""",
        default=None,
    )
    pas_de_solution_initiale: Optional[bool] = Field(
        description=r"""Resolution isn\'t initialized with the solution x""",
        default=None,
    )
    ascii: Optional[bool] = Field(description=r"""Ascii files""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "fichier_secmem": [],
        "fichier_matrice": [],
        "fichier_solution": [],
        "nb_test": [],
        "impr": [],
        "solveur": [],
        "fichier_solveur": [],
        "genere_fichier_solveur": [],
        "seuil_verification": [],
        "pas_de_solution_initiale": [],
        "ascii": [],
    }


################################################################


class Optimal(Solveur_sys_base):
    r"""
    Optimal is a solver which tests several solvers of the previous list to choose the fastest
    one for the considered linear system.
    """

    seuil: float = Field(description=r"""Convergence threshold""", default=0.0)
    impr: Optional[bool] = Field(
        description=r"""To print the convergency of the fastest solver""", default=None
    )
    quiet: Optional[bool] = Field(
        description=r"""To disable printing of information""", default=None
    )
    save_matrice: Optional[bool] = Field(
        description=r"""To save the linear system (A, x, B) into a file""", default=None
    )
    frequence_recalc: Optional[int] = Field(
        description=r"""To set a time step period (by default, 100) for re-checking the fatest solver""",
        default=None,
    )
    nom_fichier_solveur: Optional[str] = Field(
        description=r"""To specify the file containing the list of the tested solvers""",
        default=None,
    )
    fichier_solveur_non_recree: Optional[bool] = Field(
        description=r"""To avoid the creation of the file containing the list""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil": [],
        "impr": [],
        "quiet": [],
        "save_matrice": ["save_matrix"],
        "frequence_recalc": [],
        "nom_fichier_solveur": [],
        "fichier_solveur_non_recree": [],
    }


################################################################


class Gmres(Solveur_sys_base):
    r"""
    Gmres method (for non symetric matrix).
    """

    impr: Optional[bool] = Field(
        description=r"""Keyword which may be used to print the convergence.""",
        default=None,
    )
    quiet: Optional[bool] = Field(
        description=r"""To disable printing of information""", default=None
    )
    seuil: Optional[float] = Field(description=r"""Convergence value.""", default=None)
    diag: Optional[bool] = Field(
        description=r"""Keyword to use diagonal preconditionner (in place of pilut that is not parallel).""",
        default=None,
    )
    nb_it_max: Optional[int] = Field(
        description=r"""Keyword to set the maximum iterations number for the Gmres.""",
        default=None,
    )
    controle_residu: Optional[Literal[0, 1]] = Field(
        description=r"""Keyword of Boolean type (by default 0). If set to 1, the convergence occurs if the residu suddenly increases.""",
        default=None,
    )
    save_matrice: Optional[bool] = Field(
        description=r"""to save the matrix in a file.""", default=None
    )
    dim_espace_krilov: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "impr": [],
        "quiet": [],
        "seuil": [],
        "diag": [],
        "nb_it_max": [],
        "controle_residu": [],
        "save_matrice": ["save_matrix"],
        "dim_espace_krilov": [],
    }


################################################################


class Integrer_champ_med(Interprete):
    r"""
    his keyword is used to calculate a flow rate from a velocity MED field read before. The
    method is either debit_total to calculate the flow rate on the whole surface, either
    integrale_en_z to calculate flow rates between z=zmin and z=zmax on nb_tranche surfaces.
    The output file indicates first the flow rate for the whole surface and then lists for
    each tranche : the height z, the surface average value, the surface area and the flow
    rate. For the debit_total method, only one tranche is considered.

    file :z Sum(u.dS)/Sum(dS) Sum(dS) Sum(u.dS)
    """

    champ_med: str = Field(description=r"""not_set""", default="")
    methode: Literal["integrale_en_z", "debit_total"] = Field(
        description=r"""to choose between the integral following z or over the entire height (debit_total corresponds to zmin=-DMAXFLOAT, ZMax=DMAXFLOAT, nb_tranche=1)""",
        default="integrale_en_z",
    )
    zmin: Optional[float] = Field(description=r"""not_set""", default=None)
    zmax: Optional[float] = Field(description=r"""not_set""", default=None)
    nb_tranche: Optional[int] = Field(description=r"""not_set""", default=None)
    fichier_sortie: Optional[str] = Field(
        description=r"""name of the output file, by default: integrale.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "champ_med": [],
        "methode": [],
        "zmin": [],
        "zmax": [],
        "nb_tranche": [],
        "fichier_sortie": [],
    }


################################################################


class Read_med(Interprete):
    r"""
    Keyword to read MED mesh files where 'domain' corresponds to the domain name, 'file'
    corresponds to the file (written in the MED format) containing the mesh named mesh_name.

    Note about naming boundaries: When reading 'file', TRUST will detect boundaries between
    domains (Raccord) when the name of the boundary begins by 'type_raccord\_'. For example, a
    boundary named type_raccord_wall in 'file' will be considered by TRUST as a boundary named
    'wall' between two domains.

    NB: To read several domains from a mesh issued from a MED file, use Read_Med to read the
    mesh then use Create_domain_from_sub_domain keyword.

    NB: If the MED file contains one or several subdomaine defined as a group of volumes,
    then Read_MED will read it and will create two files domain_name_ssz.geo and
    domain_name_ssz_par.geo defining the subdomaines for sequential and/or parallel
    calculations. These subdomaines will be read in sequential in the datafile by including
    (after Read_Med keyword) something like:

    Read_Med ....

    Read_file domain_name_ssz.geo ;

    During the parallel calculation, you will include something:

    Scatter { ... }

    Read_file domain_name_ssz_par.geo ;
    """

    convertalltopoly: Optional[bool] = Field(
        description=r"""Option to convert mesh with mixed cells into polyhedral/polygonal cells""",
        default=None,
    )
    domain: str = Field(description=r"""Corresponds to the domain name.""", default="")
    file: str = Field(
        description=r"""File (written in the MED format, with extension '.med') containing the mesh""",
        default="",
    )
    mesh: Optional[str] = Field(
        description=r"""Name of the mesh in med file. If not specified, the first mesh will be read.""",
        default=None,
    )
    exclude_groups: Optional[List[str]] = Field(
        description=r"""List of face groups to skip in the MED file.""", default=None
    )
    include_additional_face_groups: Optional[List[str]] = Field(
        description=r"""List of face groups to read and register in the MED file.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["lire_med", "read_med_64"],
        "convertalltopoly": [],
        "domain": ["domaine"],
        "file": ["fichier"],
        "mesh": ["maillage"],
        "exclude_groups": ["exclure_groupes"],
        "include_additional_face_groups": ["inclure_groupes_faces_additionnels"],
    }


################################################################


class Champ_fonc_med(Field_base):
    r"""
    Field to read a data field in a MED-format file .med at a specified time. It is very
    useful, for example, to resume a calculation with a new or refined geometry. The field
    post-processed on the new geometry at med format is used as initial condition for the
    resume.
    """

    use_existing_domain: Optional[bool] = Field(
        description=r"""whether to optimize the field loading by indicating that the field is supported by the same mesh that was initially loaded as the domain""",
        default=None,
    )
    last_time: Optional[bool] = Field(
        description=r"""to use the last time of the MED file instead of the specified time. Mutually exclusive with 'time' parameter.""",
        default=None,
    )
    decoup: Optional[str] = Field(
        description=r"""specify a partition file.""", default=None
    )
    mesh: Optional[str] = Field(
        description=r"""Name of the mesh supporting the field. This is the name of the mesh in the MED file, and if this mesh was also used to create the TRUST domain, loading can be optimized with option 'use_existing_domain'.""",
        default=None,
    )
    domain: str = Field(
        description=r"""Name of the domain supporting the field. This is the name of the mesh in the MED file, and if this mesh was also used to create the TRUST domain, loading can be optimized with option 'use_existing_domain'.""",
        default="",
    )
    file: str = Field(description=r"""Name of the .med file.""", default="")
    field: str = Field(description=r"""Name of field to load.""", default="")
    loc: Optional[Literal["som", "elem"]] = Field(
        description=r"""To indicate where the field is localised. Default to 'elem'.""",
        default=None,
    )
    time: Optional[float] = Field(
        description=r"""Timestep to load from the MED file. Mutually exclusive with 'last_time' flag.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "use_existing_domain": [],
        "last_time": [],
        "decoup": [],
        "mesh": [],
        "domain": [],
        "file": [],
        "field": [],
        "loc": [],
        "time": [],
    }


################################################################


class Champ_fonc_med_table_temps(Champ_fonc_med):
    r"""
    Field defined as a fixed spatial shape scaled by a temporal coefficient
    """

    table_temps: Optional[str] = Field(
        description=r"""Table containing the temporal coefficient used to scale the field""",
        default=None,
    )
    table_temps_lue: Optional[str] = Field(
        description=r"""Name of the file containing the values of the temporal coefficient used to scale the field""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "table_temps": [],
        "table_temps_lue": [],
        "use_existing_domain": [],
        "last_time": [],
        "decoup": [],
        "mesh": [],
        "domain": [],
        "file": [],
        "field": [],
        "loc": [],
        "time": [],
    }


################################################################


class Champ_fonc_med_tabule(Champ_fonc_med):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "use_existing_domain": [],
        "last_time": [],
        "decoup": [],
        "mesh": [],
        "domain": [],
        "file": [],
        "field": [],
        "loc": [],
        "time": [],
    }


################################################################


class List_info_med(Listobj):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Pb_post(Pb_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Info_med(Objet_lecture):
    r"""
    not_set
    """

    file_med: str = Field(description=r"""Name of the MED file.""", default="")
    domaine: str = Field(description=r"""Name of domain.""", default="")
    pb_post: Pb_post = Field(
        description=r"""not_set""", default_factory=lambda: eval("Pb_post()")
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "file_med": [],
        "domaine": [],
        "pb_post": [],
    }


################################################################


class Pbc_med(Pb_gen_base):
    r"""
    Allows to read med files and post-process them.
    """

    list_info_med: Annotated[List[Info_med], "List_info_med"] = Field(
        description=r"""not_set""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "list_info_med": []}


################################################################


class Scattermed(Scatter):
    r"""
    This keyword will read the partition of the domain_name domain into a the MED format files
    file.med created by Medsplitter.
    """

    _synonyms: ClassVar[dict] = {None: [], "file": [], "domaine": []}


################################################################


class Champ_front_med(Front_field_base):
    r"""
    Field allowing the loading of a boundary condition from a MED file using Champ_fonc_med
    """

    champ_fonc_med: Field_base = Field(
        description=r"""a champ_fonc_med loading the values of the unknown on a domain boundary""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "champ_fonc_med": []}


################################################################


class Diffusion_negligeable(Diffusion_deriv):
    r"""
    the diffusivity will not taken in count
    """

    _synonyms: ClassVar[dict] = {None: ["negligeable"]}


################################################################


class Diffusion_option(Diffusion_deriv):
    r"""
    not_set
    """

    bloc_lecture: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: ["option"], "bloc_lecture": []}


################################################################


class Convection_negligeable(Convection_deriv):
    r"""
    For VDF and VEF discretizations. Suppresses the convection operator.
    """

    _synonyms: ClassVar[dict] = {None: ["negligeable"]}


################################################################


class Convection_amont(Convection_deriv):
    r"""
    Keyword for upwind scheme for VDF or VEF discretizations. In VEF discretization equivalent
    to generic amont for TRUST version 1.5 or later. The previous upwind scheme can be used
    with the obsolete in future amont_old keyword.
    """

    _synonyms: ClassVar[dict] = {None: ["amont"]}


################################################################


class Convection_centre(Convection_deriv):
    r"""
    For VDF and VEF discretizations.
    """

    _synonyms: ClassVar[dict] = {None: ["centre"]}


################################################################


class Convection_centre4(Convection_deriv):
    r"""
    For VDF and VEF discretizations.
    """

    _synonyms: ClassVar[dict] = {None: ["centre4"]}


################################################################


class Champ_fonc_interp(Champ_don_base):
    r"""
    Field that is interpolated from a distant domain via MEDCoupling (remapper).
    """

    nom_champ: str = Field(
        description=r"""Name of the field (for example: temperature).""", default=""
    )
    pb_loc: str = Field(description=r"""Name of the local problem.""", default="")
    pb_dist: str = Field(description=r"""Name of the distant problem.""", default="")
    dom_loc: Optional[str] = Field(
        description=r"""Name of the local domain.""", default=None
    )
    dom_dist: Optional[str] = Field(
        description=r"""Name of the distant domain.""", default=None
    )
    default_value: Optional[str] = Field(
        description=r"""Name of the distant domain.""", default=None
    )
    nature: str = Field(
        description=r"""Nature of the field (knowledge from MEDCoupling is required; IntensiveMaximum, IntensiveConservation, ...).""",
        default="",
    )
    use_overlapdec: Optional[str] = Field(
        description=r"""Nature of the field (knowledge from MEDCoupling is required; IntensiveMaximum, IntensiveConservation, ...).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "nom_champ": [],
        "pb_loc": [],
        "pb_dist": [],
        "dom_loc": [],
        "dom_dist": [],
        "default_value": [],
        "nature": [],
        "use_overlapdec": [],
    }


################################################################


class Champ_post_operateur_eqn(Champ_post_de_champs_post):
    r"""
    Post-process equation operators/sources
    """

    numero_source: Optional[int] = Field(
        description=r"""the source to be post-processed (its number). If you have only one source term, numero_source will correspond to 0 if you want to post-process that unique source""",
        default=None,
    )
    numero_op: Optional[int] = Field(
        description=r"""numero_op will be 0 (diffusive operator) or 1 (convective operator) or 2 (gradient operator) or 3 (divergence operator).""",
        default=None,
    )
    numero_masse: Optional[int] = Field(
        description=r"""numero_masse will be 0 for the mass equation operator in Pb_multiphase.""",
        default=None,
    )
    sans_solveur_masse: Optional[bool] = Field(description=r"""not_set""", default=None)
    compo: Optional[int] = Field(
        description=r"""If you want to post-process only one component of a vector field, you can specify the number of the component after compo keyword. By default, it is set to -1 which means that all the components will be post-processed. This feature is not available in VDF disretization.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["operateur_eqn"],
        "numero_source": [],
        "numero_op": [],
        "numero_masse": [],
        "sans_solveur_masse": [],
        "compo": [],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Schema_implicite_base(Schema_temps_base):
    r"""
    Basic class for implicite time scheme.
    """

    max_iter_implicite: Optional[int] = Field(
        description=r"""Maximum number of iterations allowed for the solver (by default 200).""",
        default=None,
    )
    solveur: Solveur_implicite_base = Field(
        description=r"""This keyword is used to designate the solver selected in the situation where the time scheme is an implicit scheme. solver is the name of the solver that allows equation diffusion and convection operators to be set as implicit terms. Keywords corresponding to this functionality are Simple (SIMPLE type algorithm), Simpler (SIMPLER type algorithm) for incompressible systems, Piso (Pressure Implicit with Split Operator), and Implicite (similar to PISO, but as it looks like a simplified solver, it will use fewer timesteps, and ICE (for PB_multiphase). But it may run faster because the pressure matrix is not re-assembled and thus provides CPU gains.  Advice: Since the 1.6.0 version, we recommend to use first the Implicite or Simple, then Piso, and at least Simpler. Because the two first give a fastest convergence (several times) than Piso and the Simpler has not been validated. It seems also than Implicite and Piso schemes give better results than the Simple scheme when the flow is not fully stationary. Thus, if the solution obtained with Simple is not stationary, it is recommended to switch to Piso or Implicite scheme.""",
        default_factory=lambda: eval("Solveur_implicite_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_adams_moulton_order_2(Schema_implicite_base):
    r"""
    not_set
    """

    facsec_max: Optional[float] = Field(
        description=r"""Maximum ratio allowed between time step and stability time returned by CFL condition. The initial ratio given by facsec keyword is changed during the calculation with the implicit scheme but it couldn\'t be higher than facsec_max value. Warning: Some implicit schemes do not permit high facsec_max, example Schema_Adams_Moulton_order_3 needs facsec=facsec_max=1.  Advice: The calculation may start with a facsec specified by the user and increased by the algorithm up to the facsec_max limit. But the user can also choose to specify a constant facsec (facsec_max will be set to facsec value then). Faster convergence has been seen and depends on the kind of calculation: -Hydraulic only or thermal hydraulic with forced convection and low coupling between velocity and temperature (Boussinesq value beta low), facsec between 20-30-Thermal hydraulic with forced convection and strong coupling between velocity and temperature (Boussinesq value beta high), facsec between 90-100 -Thermohydralic with natural convection, facsec around 300 -Conduction only, facsec can be set to a very high value (1e8) as if the scheme was unconditionally stableThese values can also be used as rule of thumb for initial facsec with a facsec_max limit higher.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "facsec_max": [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Euler_scheme(Schema_temps_base):
    r"""
    This is the Euler explicit scheme.
    """

    _synonyms: ClassVar[dict] = {
        None: ["schema_euler_explicite", "scheme_euler_explicit"],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Leap_frog(Schema_temps_base):
    r"""
    This is the leap-frog scheme.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_euler_implicite(Schema_implicite_base):
    r"""
    This is the Euler implicit scheme.
    """

    facsec_max: Optional[float] = Field(
        description=r"""For old syntax, see the complete parameters of facsec for details""",
        default=None,
    )
    resolution_monolithique: Optional[Bloc_lecture] = Field(
        description=r"""Activate monolithic resolution for coupled problems. Solves together the equations corresponding to the application domains in the given order. All aplication domains of the coupled equations must be given to determine the order of resolution. If the monolithic solving is not wanted for a specific application domain, an underscore can be added as prefix. For example, resolution_monolithique { dom1 { dom2 dom3 } _dom4 } will solve in a single matrix the equations having dom1 as application domain, then the equations having dom2 or dom3 as application domain in a single matrix, then the equations having dom4 as application domain in a sequential way (not in a single matrix).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["scheme_euler_implicit"],
        "facsec_max": [],
        "resolution_monolithique": [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Facsec(Interprete):
    r"""
    To parameter the safety factor for the time step during the simulation.
    """

    facsec_ini: Optional[float] = Field(
        description=r"""Initial facsec taken into account at the beginning of the simulation.""",
        default=None,
    )
    facsec_max: Optional[float] = Field(
        description=r"""Maximum ratio allowed between time step and stability time returned by CFL condition. The initial ratio given by facsec keyword is changed during the calculation with the implicit scheme but it couldn\'t be higher than facsec_max value. Warning: Some implicit schemes do not permit high facsec_max, example Schema_Adams_Moulton_order_3 needs facsec=facsec_max=1.  Advice: The calculation may start with a facsec specified by the user and increased by the algorithm up to the facsec_max limit. But the user can also choose to specify a constant facsec (facsec_max will be set to facsec value then). Faster convergence has been seen and depends on the kind of calculation: -Hydraulic only or thermal hydraulic with forced convection and low coupling between velocity and temperature (Boussinesq value beta low), facsec between 20-30-Thermal hydraulic with forced convection and strong coupling between velocity and temperature (Boussinesq value beta high), facsec between 90-100 -Thermohydralic with natural convection, facsec around 300 -Conduction only, facsec can be set to a very high value (1e8) as if the scheme was unconditionally stableThese values can also be used as rule of thumb for initial facsec with a facsec_max limit higher.""",
        default=None,
    )
    rapport_residus: Optional[float] = Field(
        description=r"""Ratio between the residual at time n and the residual at time n+1 above which the facsec is increased by multiplying by sqrt(rapport_residus) (1.2 by default).""",
        default=None,
    )
    nb_ite_sans_accel_max: Optional[int] = Field(
        description=r"""Maximum number of iterations without facsec increases (20000 by default): if facsec does not increase with the previous condition (ration between 2 consecutive residuals too high), we increase it by force after nb_ite_sans_accel_max iterations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "facsec_ini": [],
        "facsec_max": [],
        "rapport_residus": [],
        "nb_ite_sans_accel_max": [],
    }


################################################################


class Schema_adams_moulton_order_3(Schema_implicite_base):
    r"""
    not_set
    """

    facsec_max: Optional[float] = Field(
        description=r"""Maximum ratio allowed between time step and stability time returned by CFL condition. The initial ratio given by facsec keyword is changed during the calculation with the implicit scheme but it couldn\'t be higher than facsec_max value. Warning: Some implicit schemes do not permit high facsec_max, example Schema_Adams_Moulton_order_3 needs facsec=facsec_max=1.  Advice: The calculation may start with a facsec specified by the user and increased by the algorithm up to the facsec_max limit. But the user can also choose to specify a constant facsec (facsec_max will be set to facsec value then). Faster convergence has been seen and depends on the kind of calculation: -Hydraulic only or thermal hydraulic with forced convection and low coupling between velocity and temperature (Boussinesq value beta low), facsec between 20-30-Thermal hydraulic with forced convection and strong coupling between velocity and temperature (Boussinesq value beta high), facsec between 90-100 -Thermohydralic with natural convection, facsec around 300 -Conduction only, facsec can be set to a very high value (1e8) as if the scheme was unconditionally stableThese values can also be used as rule of thumb for initial facsec with a facsec_max limit higher.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "facsec_max": [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_2(Schema_temps_base):
    r"""
    This is a low-storage Runge-Kutta scheme of second order that uses 2 integration points.
    The method is presented by Williamson (case 1) in
    https://www.sciencedirect.com/science/article/pii/0021999180900339
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_4(Schema_temps_base):
    r"""
    This is a low-storage Runge-Kutta scheme of fourth order that uses 3 integration points.
    The method is presented by Williamson (case 17) in
    https://www.sciencedirect.com/science/article/pii/0021999180900339
    """

    _synonyms: ClassVar[dict] = {
        None: ["runge_kutta_ordre_4_d3p"],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_predictor_corrector(Schema_temps_base):
    r"""
    This is the predictor-corrector scheme (second order). It is more accurate and economic
    than MacCormack scheme. It gives best results with a second ordre convective scheme like
    quick, centre (VDF).
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_adams_bashforth_order_2(Schema_temps_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_adams_bashforth_order_3(Schema_temps_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_rationnel_ordre_2(Schema_temps_base):
    r"""
    This is the Runge-Kutta rational scheme of second order. The method is described in the
    note: Wambeck - Rational Runge-Kutta methods for solving systems of ordinary differential
    equations, at the link: https://link.springer.com/article/10.1007/BF02252381. Although
    rational methods require more computational work than linear ones, they can have some
    other properties, such as a stable behaviour with explicitness, which make them
    preferable. The CFD application of this RRK2 scheme is described in the note:
    https://link.springer.com/content/pdf/10.1007\%2F3-540-13917-6_112.pdf.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_2_classique(Schema_temps_base):
    r"""
    This is a classical Runge-Kutta scheme of second order that uses 2 integration points.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_3_classique(Schema_temps_base):
    r"""
    This is a classical Runge-Kutta scheme of third order that uses 3 integration points.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_4_classique(Schema_temps_base):
    r"""
    This is a classical Runge-Kutta scheme of fourth order that uses 4 integration points.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Runge_kutta_ordre_4_classique_3_8(Schema_temps_base):
    r"""
    This is a classical Runge-Kutta scheme of fourth order that uses 4 integration points and
    the 3/8 rule.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Loi_fermeture_base(Objet_u):
    r"""
    Class for appends fermeture to problem
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Loi_fermeture_test(Loi_fermeture_base):
    r"""
    Loi for test only
    """

    coef: Optional[float] = Field(description=r"""coefficient""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "coef": []}


################################################################


class Loi_horaire(Objet_u):
    r"""
    to define the movement with a time-dependant law for the solid interface.
    """

    position: List[str] = Field(
        description=r"""Vecteur position""", default_factory=list
    )
    vitesse: List[str] = Field(description=r"""Vecteur vitesse""", default_factory=list)
    rotation: Optional[List[str]] = Field(
        description=r"""Matrice de passage""", default=None
    )
    derivee_rotation: Optional[List[str]] = Field(
        description=r"""Derivee matrice de passage""", default=None
    )
    verification_derivee: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    impr: Optional[int] = Field(
        description=r"""Whether to print output""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "position": [],
        "vitesse": [],
        "rotation": [],
        "derivee_rotation": [],
        "verification_derivee": [],
        "impr": [],
    }


################################################################


class Debog(Interprete):
    r"""
    Class to debug some differences between two TRUST versions on a same data file.

    If you want to compare the results of the same code in sequential and parallel
    calculation, first run (mode=0) in sequential mode (the files fichier1 and fichier2 will
    be written first) then the second run in parallel calculation (mode=1).

    During the first run (mode=0), it prints into the file DEBOG, values at different points
    of the code thanks to the C++ instruction call. see for example in
    Kernel/Framework/Resoudre.cpp file the instruction: Debog::verifier(msg,value); Where msg
    is a string and value may be a double, an integer or an array.

    During the second run (mode=1), it prints into a file Err_Debog.dbg the same messages
    than in the DEBOG file and checks if the differences between results from both codes are
    less than a given value (error). If not, it prints Ok else show the differences and the
    lines where it occured.
    """

    pb: str = Field(description=r"""Name of the problem to debug.""", default="")
    fichier1: str = Field(
        description=r"""Name of the file where domain will be written in sequential calculation.""",
        default="",
    )
    fichier2: str = Field(
        description=r"""Name of the file where faces will be written in sequential calculation.""",
        default="",
    )
    seuil: float = Field(
        description=r"""Minimal value (by default 1.e-20) for the differences between the two codes.""",
        default=0.0,
    )
    mode: int = Field(
        description=r"""By default -1 (nothing is written in the different files), you will set 0 for the sequential run, and 1 for the parallel run.""",
        default=0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "pb": [],
        "fichier1": ["file1"],
        "fichier2": ["file2"],
        "seuil": [],
        "mode": [],
    }


################################################################


class Testeur(Interprete):
    r"""
    not_set
    """

    data: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "data": []}


################################################################


class Problem_read_generic(Pb_base):
    r"""
    The probleme_read_generic differs rom the rest of the TRUST code : The problem does not
    state the number of equations that are enclosed in the problem. As the list of equations
    to be solved in the generic read problem is declared in the data file and not pre-defined
    in the structure of the problem, each equation has to be distinctively associated with the
    problem with the Associate keyword.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Discretize(Interprete):
    r"""
    Keyword to discretise a problem problem_name according to the discretization dis.

    IMPORTANT: A number of objects must be already associated (a domain, time scheme, central
    object) prior to invoking the Discretize (Discretiser) keyword. The physical properties of
    this central object must also have been read.
    """

    problem_name: str = Field(description=r"""Name of problem.""", default="")
    dis: str = Field(description=r"""Name of the discretization object.""", default="")
    _synonyms: ClassVar[dict] = {None: ["discretiser"], "problem_name": [], "dis": []}


################################################################


class Source_generique(Source_base):
    r"""
    to define a source term depending on some discrete fields of the problem and (or) analytic
    expression. It is expressed by the way of a generic field usually used for post-
    processing.
    """

    champ: Champ_generique_base = Field(
        description=r"""the source field""",
        default_factory=lambda: eval("Champ_generique_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "champ": []}


################################################################


class Dt_calc_dt_calc(Dt_start):
    r"""
    The time step at first iteration is calculated in agreement with CFL condition.
    """

    _synonyms: ClassVar[dict] = {None: ["dt_calc"]}


################################################################


class Dt_calc_dt_min(Dt_start):
    r"""
    The first iteration is based on dt_min.
    """

    _synonyms: ClassVar[dict] = {None: ["dt_min"]}


################################################################


class Dt_calc_dt_fixe(Dt_start):
    r"""
    The first time step is fixed by the user (recommended when resuming calculation with Crank
    Nicholson temporal scheme to ensure continuity).
    """

    value: float = Field(description=r"""first time step.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["dt_fixe"], "value": []}


################################################################


class Champ_parametrique(Champ_don_base):
    r"""
    Parametric field
    """

    fichier: str = Field(description=r"""Filename where fields are read""", default="")
    _synonyms: ClassVar[dict] = {None: [], "fichier": []}


################################################################


class Solve(Interprete):
    r"""
    Interpretor to start calculation with TRUST.
    """

    pb: str = Field(description=r"""Name of problem to be solved.""", default="")
    _synonyms: ClassVar[dict] = {None: ["resoudre"], "pb": []}


################################################################


class Champ_front_parametrique(Front_field_base):
    r"""
    Parametric boundary field
    """

    fichier: str = Field(
        description=r"""Filename where boundary fields are read""", default=""
    )
    _synonyms: ClassVar[dict] = {None: [], "fichier": []}


################################################################


class Coarsen_operators(Listobj):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Coarsen_operator_uniform(Objet_lecture):
    r"""
    Object defining the uniform coarsening process of the given grid in IJK discretization
    """

    coarsen_operator_uniform: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    aco: Literal["{"] = Field(description=r"""opening curly brace""", default="{")
    coarsen_i: Optional[Literal["coarsen_i"]] = Field(
        description=r"""not_set""", default=None
    )
    coarsen_i_val: Optional[int] = Field(
        description=r"""Integer indicating the number by which we will divide the number of elements in the I direction (in order to obtain a coarser grid)""",
        default=None,
    )
    coarsen_j: Optional[Literal["coarsen_j"]] = Field(
        description=r"""not_set""", default=None
    )
    coarsen_j_val: Optional[int] = Field(
        description=r"""Integer indicating the number by which we will divide the number of elements in the J direction (in order to obtain a coarser grid)""",
        default=None,
    )
    coarsen_k: Optional[Literal["coarsen_k"]] = Field(
        description=r"""not_set""", default=None
    )
    coarsen_k_val: Optional[int] = Field(
        description=r"""Integer indicating the number by which we will divide the number of elements in the K direction (in order to obtain a coarser grid)""",
        default=None,
    )
    acof: Literal["}"] = Field(description=r"""closing curly brace""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "coarsen_operator_uniform": [],
        "aco": [],
        "coarsen_i": [],
        "coarsen_i_val": [],
        "coarsen_j": [],
        "coarsen_j_val": [],
        "coarsen_k": [],
        "coarsen_k_val": [],
        "acof": [],
    }


################################################################


class Multigrid_solver(Interprete):
    r"""
    Object defining a multigrid solver in IJK discretization
    """

    coarsen_operators: Optional[
        Annotated[List[Coarsen_operator_uniform], "Coarsen_operators"]
    ] = Field(description=r"""not_set""", default=None)
    ghost_size: Optional[int] = Field(
        description=r"""Number of ghost cells known by each processor in each of the three directions""",
        default=None,
    )
    relax_jacobi: Optional[List[float]] = Field(
        description=r"""Parameter between 0 and 1 that will be used in the Jacobi method to solve equation on each grid. Should be around 0.7""",
        default=None,
    )
    pre_smooth_steps: Optional[List[int]] = Field(
        description=r"""First integer of the list indicates the numbers of integers that has to be read next. Following integers define the numbers of iterations done before solving the equation on each grid. For example, 2 7 8 means that we have a list of 2 integers, the first one tells us to perform 7 pre-smooth steps on the first grid, the second one tells us to perform 8 pre-smooth steps on the second grid. If there are more than 2 grids in the solver, then the remaining ones will have as many pre-smooth steps as the last mentionned number (here, 8)""",
        default=None,
    )
    smooth_steps: Optional[List[int]] = Field(
        description=r"""First integer of the list indicates the numbers of integers that has to be read next. Following integers define the numbers of iterations done after solving the equation on each grid. Same behavior as pre_smooth_steps""",
        default=None,
    )
    nb_full_mg_steps: Optional[List[int]] = Field(
        description=r"""Number of multigrid iterations at each level""", default=None
    )
    solveur_grossier: Optional[Solveur_sys_base] = Field(
        description=r"""Name of the iterative solver that will be used to solve the system on the coarsest grid. This resolution must be more precise than the ones occurring on the fine grids. The threshold of this solver must therefore be lower than seuil defined above.""",
        default=None,
    )
    seuil: Optional[float] = Field(
        description=r"""Define an upper bound on the norm of the final residue (i.e. the one obtained after applying the multigrid solver). With hybrid precision, as long as we have not obtained a residue whose norm is lower than the imposed threshold, we keep applying the solver""",
        default=None,
    )
    impr: Optional[bool] = Field(
        description=r"""Flag to display some info on the resolution on eahc grid""",
        default=None,
    )
    solver_precision: Optional[Literal["mixed", "double"]] = Field(
        description=r"""Precision with which the variables at stake during the resolution of the system will be stored. We can have a simple or double precision or both. In the case of a hybrid precision, the multigrid solver is launched in simple precision, but the residual is calculated in double precision.""",
        default=None,
    )
    iterations_mixed_solver: Optional[int] = Field(
        description=r"""Define the maximum number of iterations in mixed precision solver""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "coarsen_operators": [],
        "ghost_size": [],
        "relax_jacobi": [],
        "pre_smooth_steps": [],
        "smooth_steps": [],
        "nb_full_mg_steps": [],
        "solveur_grossier": [],
        "seuil": [],
        "impr": [],
        "solver_precision": [],
        "iterations_mixed_solver": [],
    }


################################################################


class Test_sse_kernels(Interprete):
    r"""
    Object to test the different kernel methods used in the multigrid solver in IJK
    discretization
    """

    nmax: Optional[int] = Field(
        description=r"""Number of tests we want to perform""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "nmax": []}


################################################################


class Ijk_grid_geometry(Domaine):
    r"""
    Object to define the grid that will represent the domain of the simulation in IJK
    discretization
    """

    perio_i: Optional[bool] = Field(
        description=r"""flag to specify the border along the I direction is periodic""",
        default=None,
    )
    perio_j: Optional[bool] = Field(
        description=r"""flag to specify the border along the J direction is periodic""",
        default=None,
    )
    perio_k: Optional[bool] = Field(
        description=r"""flag to specify the border along the K direction is periodic""",
        default=None,
    )
    nbelem_i: Optional[int] = Field(
        description=r"""the number of elements of the grid in the I direction""",
        default=None,
    )
    nbelem_j: Optional[int] = Field(
        description=r"""the number of elements of the grid in the J direction""",
        default=None,
    )
    nbelem_k: Optional[int] = Field(
        description=r"""the number of elements of the grid in the K direction""",
        default=None,
    )
    uniform_domain_size_i: Optional[float] = Field(
        description=r"""the size of the elements along the I direction""", default=None
    )
    uniform_domain_size_j: Optional[float] = Field(
        description=r"""the size of the elements along the J direction""", default=None
    )
    uniform_domain_size_k: Optional[float] = Field(
        description=r"""the size of the elements along the K direction""", default=None
    )
    origin_i: Optional[float] = Field(
        description=r"""I-coordinate of the origin of the grid""", default=None
    )
    origin_j: Optional[float] = Field(
        description=r"""J-coordinate of the origin of the grid""", default=None
    )
    origin_k: Optional[float] = Field(
        description=r"""K-coordinate of the origin of the grid""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "perio_i": [],
        "perio_j": [],
        "perio_k": [],
        "nbelem_i": [],
        "nbelem_j": [],
        "nbelem_k": [],
        "uniform_domain_size_i": [],
        "uniform_domain_size_j": [],
        "uniform_domain_size_k": [],
        "origin_i": [],
        "origin_j": [],
        "origin_k": [],
    }


################################################################


class Parallel_io_parameters(Interprete):
    r"""
    Object to handle parallel files in IJK discretization
    """

    block_size_bytes: Optional[int] = Field(
        description=r"""File writes will be performed by chunks of this size (in bytes). This parameter will not be taken into account if block_size_megabytes has been defined""",
        default=None,
    )
    block_size_megabytes: Optional[int] = Field(
        description=r"""File writes will be performed by chunks of this size (in megabytes). The size should be a multiple of the GPFS block size or lustre stripping size (typically several megabytes)""",
        default=None,
    )
    writing_processes: Optional[int] = Field(
        description=r"""This is the number of processes that will write concurrently to the file system (this must be set according to the capacity of the filesystem, set to 1 on small computers, can be up to 64 or 128 on very large systems).""",
        default=None,
    )
    bench_ijk_splitting_write: Optional[str] = Field(
        description=r"""Name of the splitting object we want to use to run a parallel write bench (optional parameter)""",
        default=None,
    )
    bench_ijk_splitting_read: Optional[str] = Field(
        description=r"""Name of the splitting object we want to use to run a parallel read bench (optional parameter)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "block_size_bytes": [],
        "block_size_megabytes": [],
        "writing_processes": [],
        "bench_ijk_splitting_write": [],
        "bench_ijk_splitting_read": [],
    }


################################################################


class Ijk_splitting(Objet_u):
    r"""
    Object to specify how the domain will be divided between processors in IJK discretization
    """

    ijk_grid_geometry: str = Field(
        description=r"""the grid that will be splitted""", default=""
    )
    nproc_i: int = Field(
        description=r"""the number of processors into which we will divide the grid following the I direction""",
        default=0,
    )
    nproc_j: int = Field(
        description=r"""the number of processors into which we will divide the grid following the J direction""",
        default=0,
    )
    nproc_k: int = Field(
        description=r"""the number of processors into which we will divide the grid following the K direction""",
        default=0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "ijk_grid_geometry": [],
        "nproc_i": [],
        "nproc_j": [],
        "nproc_k": [],
    }


################################################################


class Longueur_melange(Mod_turb_hyd_ss_maille):
    r"""
    This model is based on mixing length modelling. For a non academic configuration,
    formulation used in the code can be expressed basically as :

    $nu_t=(Kappa.y)^2$.dU/dy

    Till a maximum distance (dmax) set by the user in the data file, y is set equal to the
    distance from the wall (dist_w) calculated previously and saved in file Wall_length.xyz.
    [see Distance_paroi keyword]

    Then (from y=dmax), y decreases as an exponential function : y=dmax*exp[-2.*(dist_w-
    dmax)/dmax]
    """

    canalx: Optional[float] = Field(
        description=r"""[height] : plane channel according to Ox direction (for the moment, formulation in the code relies on fixed heigh : H=2).""",
        default=None,
    )
    tuyauz: Optional[float] = Field(
        description=r"""[diameter] : pipe according to Oz direction (for the moment, formulation in the code relies on fixed diameter : D=2).""",
        default=None,
    )
    verif_dparoi: Optional[str] = Field(description=r"""not_set""", default=None)
    dmax: Optional[float] = Field(description=r"""Maximum distance.""", default=None)
    fichier: Optional[str] = Field(description=r"""not_set""", default=None)
    fichier_ecriture_k_eps: Optional[str] = Field(
        description=r"""When a resume with k-epsilon model is envisaged, this keyword allows to generate external MED-format file with evaluation of k and epsilon quantities (based on eddy turbulent viscosity and turbulent characteristic length returned by mixing length model). The frequency of the MED file print is set equal to dt_impr_ustar. Moreover, k-eps MED field is automatically saved at the last time step. MED file is then used for resuming a K-Epsilon calculation with the Champ_Fonc_Med keyword.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "canalx": [],
        "tuyauz": [],
        "verif_dparoi": [],
        "dmax": [],
        "fichier": [],
        "fichier_ecriture_k_eps": [],
        "formulation_a_nb_points": [],
        "longueur_maille": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Source_th_tdivu(Source_base):
    r"""
    This term source is dedicated for any scalar (called T) transport. Coupled with upwind
    (amont) or muscl scheme, this term gives for final expression of convection :
    div(U.T)-T.div (U)=U.grad(T) This ensures, in incompressible flow when divergence free is
    badly resolved, to stay in a better way in the physical boundaries.

    Warning: Only available in VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Source_qdm_lambdaup(Source_base):
    r"""
    This source term is a dissipative term which is intended to minimise the energy associated
    to non-conformscales u\' (responsible for spurious oscillations in some cases). The
    equation for these scales can be seen as: du\'/dt= -lambda. u\' + grad P\' where -lambda.
    u\' represents the dissipative term, with lambda = a/Delta t For Crank-Nicholson temporal
    scheme, recommended value for a is 2.

    Remark : This method requires to define a filtering operator.
    """

    lambda_: float = Field(description=r"""value of lambda""", default=0.0)
    lambda_min: Optional[float] = Field(
        description=r"""value of lambda_min""", default=None
    )
    lambda_max: Optional[float] = Field(
        description=r"""value of lambda_max""", default=None
    )
    ubar_umprim_cible: Optional[float] = Field(
        description=r"""value of ubar_umprim_cible""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "lambda_": ["lambda_u", "lambda"],
        "lambda_min": [],
        "lambda_max": [],
        "ubar_umprim_cible": [],
    }


################################################################


class Perte_charge_circulaire(Source_base):
    r"""
    New pressure loss.
    """

    lambda_: str = Field(
        description=r"""Function f(Re_tot, Re_long, t, x, y, z) for loss coefficient in the longitudinal direction""",
        default="",
    )
    diam_hydr: Champ_don_base = Field(
        description=r"""Hydraulic diameter value.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    sous_zone: Optional[str] = Field(
        description=r"""Optional sub-area where pressure loss applies.""", default=None
    )
    lambda_ortho: Optional[str] = Field(
        description=r"""function: Function f(Re_tot, Re_ortho, t, x, y, z) for loss coefficient in transverse direction""",
        default=None,
    )
    diam_hydr_ortho: Champ_don_base = Field(
        description=r"""Transverse hydraulic diameter value.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    direction: Champ_don_base = Field(
        description=r"""Field which indicates the direction of the pressure loss.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "lambda_": ["lambda_u", "lambda"],
        "diam_hydr": [],
        "sous_zone": [],
        "lambda_ortho": [],
        "diam_hydr_ortho": [],
        "direction": [],
    }


################################################################


class Perte_charge_anisotrope(Source_base):
    r"""
    Anisotropic pressure loss.
    """

    lambda_: str = Field(
        description=r"""Function for loss coefficient which may be Reynolds dependant (Ex: 64/Re).""",
        default="",
    )
    lambda_ortho: str = Field(
        description=r"""Function for loss coefficient in transverse direction which may be Reynolds dependant (Ex: 64/Re).""",
        default="",
    )
    diam_hydr: Champ_don_base = Field(
        description=r"""Hydraulic diameter value.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    direction: Champ_don_base = Field(
        description=r"""Field which indicates the direction of the pressure loss.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    sous_zone: Optional[str] = Field(
        description=r"""Optional sub-area where pressure loss applies.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "lambda_": ["lambda_u", "lambda"],
        "lambda_ortho": [],
        "diam_hydr": [],
        "direction": [],
        "sous_zone": [],
    }


################################################################


class Perte_charge_directionnelle(Source_base):
    r"""
    Directional pressure loss (available in VEF and PolyMAC).
    """

    lambda_: str = Field(
        description=r"""Function for loss coefficient which may be Reynolds dependant (Ex: 64/Re).""",
        default="",
    )
    diam_hydr: Champ_don_base = Field(
        description=r"""Hydraulic diameter value.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    direction: Champ_don_base = Field(
        description=r"""Field which indicates the direction of the pressure loss.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    sous_zone: Optional[str] = Field(
        description=r"""Optional sub-area where pressure loss applies.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "lambda_": ["lambda_u", "lambda"],
        "diam_hydr": [],
        "direction": [],
        "sous_zone": [],
    }


################################################################


class Perte_charge_isotrope(Source_base):
    r"""
    Isotropic pressure loss (available in VEF and PolyMAC).
    """

    lambda_: str = Field(
        description=r"""Function for loss coefficient which may be Reynolds dependant (Ex: 64/Re).""",
        default="",
    )
    diam_hydr: Champ_don_base = Field(
        description=r"""Hydraulic diameter value.""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    sous_zone: Optional[str] = Field(
        description=r"""Optional sub-area where pressure loss applies.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "lambda_": ["lambda_u", "lambda"],
        "diam_hydr": [],
        "sous_zone": [],
    }


################################################################


class Frontiere_ouverte_gradient_pression_libre_vef(Neumann):
    r"""
    Class for outlet boundary condition in VEF like Orlansky. There is no reference for
    pressure for theses boundary conditions so it is better to add pressure condition (with
    Frontiere_ouverte_pression_imposee) on one or two cells (for symmetry in a channel) of the
    boundary where Orlansky conditions are imposed.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Frontiere_ouverte_gradient_pression_impose_vefprep1b(
    Frontiere_ouverte_gradient_pression_impose
):
    r"""
    Keyword for an outlet boundary condition in VEF P1B/P1NC on the gradient of the pressure.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_echange_contact_correlation_vef(Condlim_base):
    r"""
    Class to define a thermohydraulic 1D model which will apply to a boundary of 2D or 3D
    domain.

    Warning : For parallel calculation, the only possible partition will be according the
    axis of the model with the keyword Tranche_geom.
    """

    dir: Optional[int] = Field(
        description=r"""Direction (0 : axis X, 1 : axis Y, 2 : axis Z) of the 1D model.""",
        default=None,
    )
    tinf: Optional[float] = Field(
        description=r"""Inlet fluid temperature of the 1D model (oC or K).""",
        default=None,
    )
    tsup: Optional[float] = Field(
        description=r"""Outlet fluid temperature of the 1D model (oC or K).""",
        default=None,
    )
    lambda_: Optional[str] = Field(
        description=r"""Thermal conductivity of the fluid (W.m-1.K-1).""", default=None
    )
    rho: Optional[str] = Field(
        description=r"""Mass density of the fluid (kg.m-3) which may be a function of the temperature T.""",
        default=None,
    )
    dt_impr: Optional[float] = Field(
        description=r"""Printing period in name_of_data_file_time.dat files of the 1D model results.""",
        default=None,
    )
    cp: Optional[float] = Field(
        description=r"""Calorific capacity value at a constant pressure of the fluid (J.kg-1.K-1).""",
        default=None,
    )
    mu: Optional[str] = Field(
        description=r"""Dynamic viscosity of the fluid (kg.m-1.s-1) which may be a function of thetemperature T.""",
        default=None,
    )
    debit: Optional[float] = Field(
        description=r"""Surface flow rate (kg.s-1.m-2) of the fluid into the channel.""",
        default=None,
    )
    n: Optional[int] = Field(
        description=r"""Number of 1D cells of the 1D mesh.""", default=None
    )
    dh: Optional[str] = Field(
        description=r"""Hydraulic diameter may be a function f(x) with x position along the 1D axis (xinf <= x <= xsup)""",
        default=None,
    )
    surface: Optional[str] = Field(
        description=r"""Section surface of the channel which may be function f(Dh,x) of the hydraulic diameter (Dh) and x position along the 1D axis (xinf <= x <= xsup)""",
        default=None,
    )
    xinf: Optional[float] = Field(
        description=r"""Position of the inlet of the 1D mesh on the axis direction.""",
        default=None,
    )
    xsup: Optional[float] = Field(
        description=r"""Position of the outlet of the 1D mesh on the axis direction.""",
        default=None,
    )
    nu: Optional[str] = Field(
        description=r"""Nusselt number which may be a function of the Reynolds number (Re) and the Prandtl number (Pr).""",
        default=None,
    )
    emissivite_pour_rayonnement_entre_deux_plaques_quasi_infinies: Optional[float] = (
        Field(
            description=r"""Coefficient of emissivity for radiation between two quasi infinite plates.""",
            default=None,
        )
    )
    reprise_correlation: Optional[bool] = Field(
        description=r"""Keyword in the case of a resuming calculation with this correlation.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dir": [],
        "tinf": [],
        "tsup": [],
        "lambda_": ["lambda"],
        "rho": [],
        "dt_impr": [],
        "cp": [],
        "mu": [],
        "debit": [],
        "n": [],
        "dh": [],
        "surface": [],
        "xinf": [],
        "xsup": [],
        "nu": [],
        "emissivite_pour_rayonnement_entre_deux_plaques_quasi_infinies": [],
        "reprise_correlation": [],
    }


################################################################


class Frontiere_ouverte_gradient_pression_libre_vefprep1b(Neumann):
    r"""
    Class for outlet boundary condition in VEF P1B/P1NC like Orlansky.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Verifiercoin_bloc(Objet_lecture):
    r"""
    not_set
    """

    read_file: Optional[str] = Field(
        description=r"""name of the *.decoupage_som file""", default=None
    )
    expert_only: Optional[bool] = Field(
        description=r"""to not check the mesh""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "read_file": ["filename", "lire_fichier"],
        "expert_only": [],
    }


################################################################


class Verifiercoin(Interprete):
    r"""
    This keyword subdivides inconsistent 2D/3D cells used with VEFPreP1B discretization. Must
    be used before the mesh is discretized.
    The Read_file option can be used only if the file.decoupage_som was previously created by
    TRUST. This option, only in 2D, reverses the common face at two cells (at least one is
    inconsistent), through the nodes opposed. In 3D, the option has no effect.

    The expert_only option deactivates, into the VEFPreP1B divergence operator, the test of
    inconsistent cells.
    """

    domain_name: str = Field(description=r"""Name of the domaine""", default="")
    bloc: Verifiercoin_bloc = Field(
        description=r"""not_set""", default_factory=lambda: eval("Verifiercoin_bloc()")
    )
    _synonyms: ClassVar[dict] = {None: [], "domain_name": ["dom"], "bloc": []}


################################################################


class Vef(Discretisation_base):
    r"""
    Finite element volume discretization (P1NC/P1-bubble element). Since the 1.5.5 version,
    several new discretizations are available thanks to the optional keyword Read. By default,
    the VEFPreP1B keyword is equivalent to the former VEFPreP1B formulation (v1.5.4 and
    sooner). P0P1 (if used with the strong formulation for imposed pressure boundary) is
    equivalent to VEFPreP1B but the convergence is slower. VEFPreP1B dis is equivalent to
    VEFPreP1B dis Read dis { P0 P1 Changement_de_base_P1Bulle 1 Cl_pression_sommet_faible 0 }
    """

    changement_de_base_p1bulle: Optional[Literal[0, 1]] = Field(
        description=r"""changement_de_base_p1bulle 1 This option may be used to have the P1NC/P0P1 formulation (value set to 0) or the P1NC/P1Bulle formulation (value set to 1, the default).""",
        default=None,
    )
    p0: Optional[bool] = Field(
        description=r"""Pressure nodes are added on element centres""", default=None
    )
    p1: Optional[bool] = Field(
        description=r"""Pressure nodes are added on vertices""", default=None
    )
    pa: Optional[bool] = Field(
        description=r"""Only available in 3D, pressure nodes are added on bones""",
        default=None,
    )
    rt: Optional[bool] = Field(
        description=r"""For P1NCP1B (in TrioCFD)""", default=None
    )
    modif_div_face_dirichlet: Optional[Literal[0, 1]] = Field(
        description=r"""This option (by default 0) is used to extend control volumes for the momentum equation.""",
        default=None,
    )
    cl_pression_sommet_faible: Optional[Literal[0, 1]] = Field(
        description=r"""This option is used to specify a strong formulation (value set to 0, the default) or a weak formulation (value set to 1) for an imposed pressure boundary condition. The first formulation converges quicker and is stable in general cases. The second formulation should be used if there are several outlet boundaries with Neumann condition (see Ecoulement_Neumann test case for example).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["vefprep1b"],
        "changement_de_base_p1bulle": [],
        "p0": [],
        "p1": [],
        "pa": [],
        "rt": [],
        "modif_div_face_dirichlet": [],
        "cl_pression_sommet_faible": [],
    }


################################################################


class Tparoi_vef(Champ_post_de_champs_post):
    r"""
    This keyword is used to post process (only for VEF discretization) the temperature field
    with a slight difference on boundaries with Neumann condition where law of the wall is
    applied on the temperature field. nom_pb is the problem name and field_name is the
    selected field name. A keyword (temperature_physique) is available to post process this
    field without using Definition_champs.
    """

    _synonyms: ClassVar[dict] = {
        None: ["champ_post_tparoi_vef"],
        "source": [],
        "sources": [],
        "nom_source": [],
        "source_reference": [],
        "sources_reference": [],
    }


################################################################


class Champ_som_lu_vef(Champ_don_base):
    r"""
    Keyword to read in a file values located at the nodes of a mesh in VEF discretization.
    """

    domain_name: str = Field(description=r"""Name of the domain.""", default="")
    dim: int = Field(description=r"""Value of the dimension of the field.""", default=0)
    tolerance: float = Field(
        description=r"""Value of the tolerance to check the coordinates of the nodes.""",
        default=0.0,
    )
    file: str = Field(
        description=r"""Name of the file.  This file has the following format:  Xi Yi Zi -> Coordinates of the node  Ui Vi Wi -> Value of the field on this node  Xi+1 Yi+1 Zi+1 -> Next point  Ui+1 Vi+1 Zi+1 -> Next value ...""",
        default="",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domain_name": [],
        "dim": [],
        "tolerance": [],
        "file": [],
    }


################################################################


class Champ_front_tangentiel_vef(Front_field_base):
    r"""
    Field to define the tangential velocity vector field standard at the boundary in VEF
    discretization.
    """

    mot: Literal["vitesse_tangentielle"] = Field(
        description=r"""Name of vector field.""", default="vitesse_tangentielle"
    )
    vit_tan: float = Field(description=r"""Vector field standard [m/s].""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "mot": [], "vit_tan": []}


################################################################


class Convection_muscl3(Convection_deriv):
    r"""
    Keyword for a scheme using a ponderation between muscl and center schemes in VEF.
    """

    alpha: Optional[float] = Field(
        description=r"""To weight the scheme centering with the factor double (between 0 (full centered) and 1 (muscl), by default 1).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["muscl3"], "alpha": []}


################################################################


class Convection_centre_old(Convection_deriv):
    r"""
    Only for VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["centre_old"]}


################################################################


class Convection_generic(Convection_deriv):
    r"""
    Keyword for generic calling of upwind and muscl convective scheme in VEF discretization.
    For muscl scheme, limiters and order for fluxes calculations have to be specified. The
    available limiters are : minmod - vanleer -vanalbada - chakravarthy - superbee, and the
    order of accuracy is 1 or 2. Note that chakravarthy is a non-symmetric limiter and
    superbee may engender results out of physical limits. By consequence, these two limiters
    are not recommended.

    Examples:

    convection { generic amont }

    convection { generic muscl minmod 1 }

    convection { generic muscl vanleer 2 }


    In case of results out of physical limits with muscl scheme (due for instance to strong
    non-conformal velocity flow field), user can redefine in data file a lower order and a
    smoother limiter, as : convection { generic muscl minmod 1 }
    """

    type: Literal["amont", "muscl", "centre"] = Field(
        description=r"""type of scheme""", default="amont"
    )
    limiteur: Optional[
        Literal["minmod", "vanleer", "vanalbada", "chakravarthy", "superbee"]
    ] = Field(description=r"""type of limiter""", default=None)
    ordre: Optional[Literal[1, 2, 3]] = Field(
        description=r"""order of accuracy""", default=None
    )
    alpha: Optional[float] = Field(description=r"""alpha""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["generic"],
        "type": [],
        "limiteur": [],
        "ordre": [],
        "alpha": [],
    }


################################################################


class Convection_quick(Convection_deriv):
    r"""
    Only for VDF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["quick"]}


################################################################


class Convection_kquick(Convection_deriv):
    r"""
    Only for VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["kquick"]}


################################################################


class Convection_amont_old(Convection_deriv):
    r"""
    Only for VEF discretization, obsolete keyword, see amont.
    """

    _synonyms: ClassVar[dict] = {None: ["amont_old"]}


################################################################


class Convection_muscl(Convection_deriv):
    r"""
    Keyword for muscl scheme in VEF discretization equivalent to generic muscl vanleer 2 for
    the 1.5 version or later. The previous muscl scheme can be used with the obsolete in
    future muscl_old keyword.
    """

    _synonyms: ClassVar[dict] = {None: ["muscl"]}


################################################################


class Convection_di_l2(Convection_deriv):
    r"""
    Only for VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["di_l2"]}


################################################################


class Convection_muscl_old(Convection_deriv):
    r"""
    Only for VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["muscl_old"]}


################################################################


class Convection_muscl_new(Convection_deriv):
    r"""
    Only for VEF discretization.
    """

    _synonyms: ClassVar[dict] = {None: ["muscl_new"]}


################################################################


class Bloc_ef(Objet_lecture):
    r"""
    not_set
    """

    mot1: Literal["transportant_bar", "transporte_bar", "filtrer_resu", "antisym"] = (
        Field(description=r"""not_set""", default="transportant_bar")
    )
    val1: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot2: Literal["transportant_bar", "transporte_bar", "filtrer_resu", "antisym"] = (
        Field(description=r"""not_set""", default="transportant_bar")
    )
    val2: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot3: Literal["transportant_bar", "transporte_bar", "filtrer_resu", "antisym"] = (
        Field(description=r"""not_set""", default="transportant_bar")
    )
    val3: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot4: Literal["transportant_bar", "transporte_bar", "filtrer_resu", "antisym"] = (
        Field(description=r"""not_set""", default="transportant_bar")
    )
    val4: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "mot1": [],
        "val1": [],
        "mot2": [],
        "val2": [],
        "mot3": [],
        "val3": [],
        "mot4": [],
        "val4": [],
    }


################################################################


class Convection_ef(Convection_deriv):
    r"""
    For VEF calculations, a centred convective scheme based on Finite Elements formulation can
    be called through the following data:


    Convection { EF transportant_bar val transporte_bar val antisym val filtrer_resu val }


    This scheme is 2nd order accuracy (and get better the property of kinetic energy
    conservation). Due to possible problems of instabilities phenomena, this scheme has to be
    coupled with stabilisation process (see Source_Qdm_lambdaup).These two last data are
    equivalent from a theoretical point of view in variationnal writing to : div(( u. grad ub
    , vb) - (u. grad vb, ub)), where vb corresponds to the filtered reference test functions.


    Remark:

    This class requires to define a filtering operator : see solveur_bar
    """

    mot1: Optional[Literal["defaut_bar"]] = Field(
        description=r"""equivalent to transportant_bar 0 transporte_bar 1 filtrer_resu 1 antisym 1""",
        default=None,
    )
    bloc_ef: Optional[Bloc_ef] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["ef"], "mot1": [], "bloc_ef": []}


################################################################


class Sous_zone_valeur(Objet_lecture):
    r"""
    Two words.
    """

    sous_zone: str = Field(description=r"""sous zone""", default="")
    valeur: float = Field(description=r"""value""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "sous_zone": [], "valeur": []}


################################################################


class Listsous_zone_valeur(Listobj):
    r"""
    List of groups of two words.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Convection_ef_stab(Convection_deriv):
    r"""
    Keyword for a VEF convective scheme.
    """

    alpha: Optional[float] = Field(
        description=r"""To weight the scheme centering with the factor double (between 0 (full centered) and 1 (mix between upwind and centered), by default 1). For scalar equation, it is adviced to use alpha=1 and for the momentum equation, alpha=0.2 is adviced.""",
        default=None,
    )
    test: Optional[int] = Field(
        description=r"""Developer option to compare old and new version of EF_stab""",
        default=None,
    )
    tdivu: Optional[bool] = Field(
        description=r"""To have the convective operator calculated as div(TU)-TdivU(=UgradT).""",
        default=None,
    )
    old: Optional[bool] = Field(
        description=r"""To use old version of EF_stab scheme (default no).""",
        default=None,
    )
    volumes_etendus: Optional[bool] = Field(
        description=r"""Option for the scheme to use the extended volumes (default, yes).""",
        default=None,
    )
    volumes_non_etendus: Optional[bool] = Field(
        description=r"""Option for the scheme to not use the extended volumes (default, no).""",
        default=None,
    )
    amont_sous_zone: Optional[str] = Field(
        description=r"""Option to degenerate EF_stab scheme into Amont (upwind) scheme in the sub zone of name sz_name. The sub zone may be located arbitrarily in the domain but the more often this option will be activated in a zone where EF_stab scheme generates instabilities as for free outlet for example.""",
        default=None,
    )
    alpha_sous_zone: Optional[
        Annotated[List[Sous_zone_valeur], "Listsous_zone_valeur"]
    ] = Field(description=r"""List of groups of two words.""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["ef_stab"],
        "alpha": [],
        "test": [],
        "tdivu": [],
        "old": [],
        "volumes_etendus": [],
        "volumes_non_etendus": [],
        "amont_sous_zone": [],
        "alpha_sous_zone": [],
    }


################################################################


class Bloc_diffusion_standard(Objet_lecture):
    r"""
    grad_Ubar 1 makes the gradient calculated through the filtered values of velocity
    (P1-conform).

    nu 1 (respectively nut 1) takes the molecular viscosity (eddy viscosity) into account in
    the velocity gradient part of the diffusion expression.

    nu_transp 1 (respectively nut_transp 1) takes the molecular viscosity (eddy viscosity)
    into account according in the TRANSPOSED velocity gradient part of the diffusion
    expression.

    filtrer_resu 1 allows to filter the resulting diffusive fluxes contribution.
    """

    mot1: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val1: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot2: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val2: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot3: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val3: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot4: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val4: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot5: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val5: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    mot6: Literal[
        "grad_ubar", "nu", "nut", "nu_transp", "nut_transp", "filtrer_resu"
    ] = Field(description=r"""not_set""", default="grad_ubar")
    val6: Literal[0, 1] = Field(description=r"""not_set""", default=0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "mot1": [],
        "val1": [],
        "mot2": [],
        "val2": [],
        "mot3": [],
        "val3": [],
        "mot4": [],
        "val4": [],
        "mot5": [],
        "val5": [],
        "mot6": [],
        "val6": [],
    }


################################################################


class Diffusion_standard(Diffusion_deriv):
    r"""
    A new keyword, intended for LES calculations, has been developed to optimise and
    parameterise each term of the diffusion operator. Remark:


    1. This class requires to define a filtering operator : see solveur_bar

    2. The former (original) version: diffusion { } -which omitted some of the term of the
    diffusion operator- can be recovered by using the following parameters in the new class :

    diffusion { standard grad_Ubar 0 nu 1 nut 1 nu_transp 0 nut_transp 1 filtrer_resu 0}.
    """

    mot1: Optional[Literal["defaut_bar"]] = Field(
        description=r"""equivalent to grad_Ubar 1 nu 1 nut 1 nu_transp 1 nut_transp 1 filtrer_resu 1""",
        default=None,
    )
    bloc_diffusion_standard: Optional[Bloc_diffusion_standard] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["standard"],
        "mot1": [],
        "bloc_diffusion_standard": [],
    }


################################################################


class Diffusion_p1ncp1b(Diffusion_deriv):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["p1ncp1b"]}


################################################################


class Difusion_p1b(Diffusion_deriv):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["p1b"]}


################################################################


class Diffusion_stab(Diffusion_deriv):
    r"""
    keyword allowing consistent and stable calculations even in case of obtuse angle meshes.
    """

    standard: Optional[int] = Field(
        description=r"""to recover the same results as calculations made by standard laminar diffusion operator. However, no stabilization technique is used and calculations may be unstable when working with obtuse angle meshes (by default 0)""",
        default=None,
    )
    info: Optional[int] = Field(
        description=r"""developer option to get the stabilizing ratio (by default 0)""",
        default=None,
    )
    new_jacobian: Optional[int] = Field(
        description=r"""when implicit time schemes are used, this option defines a new jacobian that may be more suitable to get stationary solutions (by default 0)""",
        default=None,
    )
    nu: Optional[int] = Field(
        description=r"""(respectively nut 1) takes the molecular viscosity (resp. eddy viscosity) into account in the velocity gradient part of the diffusion expression (by default nu=1 and nut=1)""",
        default=None,
    )
    nut: Optional[int] = Field(description=r"""not_set""", default=None)
    nu_transp: Optional[int] = Field(
        description=r"""(respectively nut_transp 1) takes the molecular viscosity (resp. eddy viscosity) into account in the transposed velocity gradient part of the diffusion expression (by default nu_transp=0 and nut_transp=1)""",
        default=None,
    )
    nut_transp: Optional[int] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["stab"],
        "standard": [],
        "info": [],
        "new_jacobian": [],
        "nu": [],
        "nut": [],
        "nu_transp": [],
        "nut_transp": [],
    }


################################################################


class Echange_couplage_thermique(Paroi_echange_global_impose):
    r"""
    Thermal coupling boundary condition
    """

    text: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    h_imp: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    himpc: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    ch: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    temperature_paroi: Optional[Field_base] = Field(
        description=r"""Temperature""", default=None
    )
    flux_paroi: Optional[Field_base] = Field(
        description=r"""Wall heat flux""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "temperature_paroi": [], "flux_paroi": []}


################################################################


class Solide(Milieu_base):
    r"""
    Solid with cp and/or rho non-uniform.
    """

    rho: Optional[Field_base] = Field(
        description=r"""Density (kg.m-3).""", default=None
    )
    cp: Optional[Field_base] = Field(
        description=r"""Specific heat (J.kg-1.K-1).""", default=None
    )
    lambda_: Optional[Field_base] = Field(
        description=r"""Conductivity (W.m-1.K-1).""", default=None
    )
    user_field: Optional[Field_base] = Field(
        description=r"""user defined field.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "rho": [],
        "cp": [],
        "lambda_": ["lambda_u", "lambda"],
        "user_field": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
    }


################################################################


class Echange_interne_global_impose(Condlim_base):
    r"""
    Internal heat exchange boundary condition with global exchange coefficient.
    """

    h_imp: str = Field(
        description=r"""Global exchange coefficient value. The global exchange coefficient value is expressed in W.m-2.K-1.""",
        default="",
    )
    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: ["paroi_echange_interne_global_impose"],
        "h_imp": [],
        "ch": [],
    }


################################################################


class Puissance_thermique(Source_base):
    r"""
    Class to define a source term corresponding to a volume power release in the energy
    equation.
    """

    ch: Field_base = Field(
        description=r"""Thermal power field type. To impose a volume power on a domain sub-area, the Champ_Uniforme_Morceaux (partly_uniform_field) type must be used.  Warning : The volume thermal power is expressed in W.m-3 in 3D (in W.m-2 in 2D). It is a power per volume unit (in a porous media, it is a power per fluid volume unit).""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Echange_interne_global_parfait(Condlim_base):
    r"""
    Internal heat exchange boundary condition with perfect (infinite) exchange coefficient.
    """

    _synonyms: ClassVar[dict] = {None: ["paroi_echange_interne_global_parfait"]}


################################################################


class Conduction(Eqn_base):
    r"""
    Heat equation.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_conduction(Pb_base):
    r"""
    Resolution of the heat equation.
    """

    solide: Optional[Solide] = Field(
        description=r"""The medium associated with the problem.""", default=None
    )
    conduction: Optional[Conduction] = Field(
        description=r"""Heat equation.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "solide": [],
        "conduction": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Option_polymac(Interprete):
    r"""
    Class of PolyMAC options.
    """

    use_osqp: Optional[bool] = Field(
        description=r"""Flag to use the old formulation of the M2 matrix provided by the OSQP library. Only useful for PolyMAC version.""",
        default=None,
    )
    maillage_vdf: Optional[bool] = Field(
        description=r"""Flag used to force the calculation of the equiv tab.""",
        default=None,
    )
    interp_ve1: Optional[bool] = Field(
        description=r"""Flag to enable a first-order face-to-element velocity interpolation. By default, it is not activated which means a second order interpolation. Only useful for PolyMAC_P0 version.""",
        default=None,
    )
    traitement_axi: Optional[bool] = Field(
        description=r"""Flag used to relax the time-step stability criterion in case of a thin slice geometry while modelling an axi-symetrical case. Only useful for PolyMAC_P0 version.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "use_osqp": [],
        "maillage_vdf": ["vdf_mesh"],
        "interp_ve1": [],
        "traitement_axi": [],
    }


################################################################


class Correction_antal(Source_base):
    r"""
    Antal correction source term for multiphase problem
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Radioactive_decay(Source_base):
    r"""
    Radioactive decay source term of the form $-\lambda_i c_i$, where $0 \leq i \leq N$, N is
    the number of component of the constituent, $c_i$ and $\lambda_i$ are the concentration
    and the decay constant of the i-th component of the constituant.
    """

    val: List[float] = Field(
        description=r"""n is the number of decay constants to read (int), and val1, val2... are the decay constants (double)""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Polymac_p0(Discretisation_base):
    r"""
    polymac_p0 discretization (previously covimac discretization compatible with pb_multi).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Polymac(Discretisation_base):
    r"""
    polymac discretization (polymac discretization that is not compatible with pb_multi).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Polymac_p0p1nc(Discretisation_base):
    r"""
    polymac_P0P1NC discretization (previously polymac discretization compatible with
    pb_multi).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Op_conv_ef_stab_polymac_face(Interprete):
    r"""
    Class Op_Conv_EF_Stab_PolyMAC_Face_PolyMAC
    """

    alpha: Optional[float] = Field(
        description=r"""parametre ajustant la stabilisation de 0 (schema centre) a 1 (schema amont)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "alpha": []}


################################################################


class Op_conv_ef_stab_polymac_p0p1nc_elem(Interprete):
    r"""
    Class Op_Conv_EF_Stab_PolyMAC_P0P1NC_Elem
    """

    alpha: Optional[float] = Field(
        description=r"""parametre ajustant la stabilisation de 0 (schema centre) a 1 (schema amont)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["op_conv_ef_stab_polymac_p0_elem"], "alpha": []}


################################################################


class Op_conv_ef_stab_polymac_p0_face(Interprete):
    r"""
    Class Op_Conv_EF_Stab_PolyMAC_P0_Face
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Op_conv_ef_stab_polymac_p0p1nc_face(Interprete):
    r"""
    Class Op_Conv_EF_Stab_PolyMAC_P0P1NC_Face
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Bloc_pdf_model(Objet_lecture):
    r"""
    not_set
    """

    eta: float = Field(description=r"""penalization coefficient""", default=0.0)
    temps_relaxation_coefficient_pdf: Optional[float] = Field(
        description=r"""time relaxation on the forcing term to help""", default=None
    )
    echelle_relaxation_coefficient_pdf: Optional[float] = Field(
        description=r"""time relaxation on the forcing term to help convergence""",
        default=None,
    )
    local: Optional[bool] = Field(
        description=r"""whether the prescribed velocity is expressed in the global or local basis""",
        default=None,
    )
    vitesse_imposee_data: Optional[Field_base] = Field(
        description=r"""Prescribed velocity as a field""", default=None
    )
    vitesse_imposee_fonction: Optional[Troismots] = Field(
        description=r"""Prescribed velocity as a set of ananlytical component""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "eta": [],
        "temps_relaxation_coefficient_pdf": [],
        "echelle_relaxation_coefficient_pdf": [],
        "local": [],
        "vitesse_imposee_data": [],
        "vitesse_imposee_fonction": [],
    }


################################################################


class Source_pdf_base(Source_base):
    r"""
    Base class of the source term for the Immersed Boundary Penalized Direct Forcing method
    (PDF)
    """

    aire: Field_base = Field(
        description=r"""volumic field: a boolean for the cell (0 or 1) indicating if the obstacle is in the cell""",
        default_factory=lambda: eval("Field_base()"),
    )
    rotation: Field_base = Field(
        description=r"""volumic field with 9 components representing the change of basis on cells (local to global). Used for rotating cases for example.""",
        default_factory=lambda: eval("Field_base()"),
    )
    transpose_rotation: Optional[bool] = Field(
        description=r"""whether to transpose the basis change matrix.""", default=None
    )
    modele: Bloc_pdf_model = Field(
        description=r"""model used for the Penalized Direct Forcing""",
        default_factory=lambda: eval("Bloc_pdf_model()"),
    )
    interpolation: Optional[Interpolation_ibm_base] = Field(
        description=r"""interpolation method""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "aire": [],
        "rotation": [],
        "transpose_rotation": [],
        "modele": [],
        "interpolation": [],
    }


################################################################


class Source_pdf(Source_pdf_base):
    r"""
    Source term for Penalised Direct Forcing (PDF) method.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "aire": [],
        "rotation": [],
        "transpose_rotation": [],
        "modele": [],
        "interpolation": [],
    }


################################################################


class Paroi_fixe(Condlim_base):
    r"""
    Keyword to designate a situation of adherence to the wall called bord (edge) (normal and
    tangential velocity at the edge is zero).
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_fixe_iso_genepi2_sans_contribution_aux_vitesses_sommets(Paroi_fixe):
    r"""
    Boundary condition to obtain iso Geneppi2, without interest
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Ef(Discretisation_base):
    r"""
    Element Finite discretization.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Convection_btd(Convection_deriv):
    r"""
    Only for EF discretization.
    """

    btd: float = Field(description=r"""not_set""", default=0.0)
    facteur: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["btd"], "btd": [], "facteur": []}


################################################################


class Convection_supg(Convection_deriv):
    r"""
    Only for EF discretization.
    """

    facteur: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["supg"], "facteur": []}


################################################################


class Convection_ale(Convection_deriv):
    r"""
    A convective scheme for ALE (Arbitrary Lagrangian-Eulerian) framework.
    """

    opconv: Bloc_convection = Field(
        description=r"""Choice between: amont and muscl  Example: convection { ALE { amont } }""",
        default_factory=lambda: eval("Bloc_convection()"),
    )
    _synonyms: ClassVar[dict] = {None: ["ale"], "opconv": []}


################################################################


class Fluide_base(Milieu_base):
    r"""
    Basic class for fluids.
    """

    indice: Optional[Field_base] = Field(
        description=r"""Refractivity of fluid.""", default=None
    )
    kappa: Optional[Field_base] = Field(
        description=r"""Absorptivity of fluid (m-1).""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Fluide_incompressible(Fluide_base):
    r"""
    Class for non-compressible fluids.
    """

    beta_th: Optional[Field_base] = Field(
        description=r"""Thermal expansion (K-1).""", default=None
    )
    mu: Optional[Field_base] = Field(
        description=r"""Dynamic viscosity (kg.m-1.s-1).""", default=None
    )
    beta_co: Optional[Field_base] = Field(
        description=r"""Volume expansion coefficient values in concentration.""",
        default=None,
    )
    rho: Optional[Field_base] = Field(
        description=r"""Density (kg.m-3).""", default=None
    )
    cp: Optional[Field_base] = Field(
        description=r"""Specific heat (J.kg-1.K-1).""", default=None
    )
    lambda_: Optional[Field_base] = Field(
        description=r"""Conductivity (W.m-1.K-1).""", default=None
    )
    porosites: Optional[Bloc_lecture] = Field(
        description=r"""Porosity (optional)""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "beta_th": [],
        "mu": [],
        "beta_co": [],
        "rho": [],
        "cp": [],
        "lambda_": ["lambda_u", "lambda"],
        "porosites": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
    }


################################################################


class Fluide_ostwald(Fluide_incompressible):
    r"""
    Non-Newtonian fluids governed by Ostwald\'s law. The law applicable to stress tensor is:

    tau=K(T)*(D:D/2)**((n-1)/2)*D Where:

    D refers to the deformation tensor

    K refers to fluid consistency (may be a function of the temperature T)

    n refers to the fluid structure index n=1 for a Newtonian fluid, n<1 for a rheofluidifier
    fluid, n>1 for a rheothickening fluid.
    """

    k: Optional[Field_base] = Field(description=r"""Fluid consistency.""", default=None)
    n: Optional[Field_base] = Field(
        description=r"""Fluid structure index.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "k": [],
        "n": [],
        "beta_th": [],
        "mu": [],
        "beta_co": [],
        "rho": [],
        "cp": [],
        "lambda_": ["lambda_u", "lambda"],
        "porosites": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
    }


################################################################


class Type_perte_charge_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_perte_charge_dp(Type_perte_charge_deriv):
    r"""
    DP field should have 3 components defining dp, dDP/dQ, Q0
    """

    dp_field: Field_base = Field(
        description=r"""the parameters of the previous formula (DP = dp + dDP/dQ * (Q - Q0)): uniform_field 3 dp dDP/dQ Q0 where Q0 is a mass flow rate (kg/s).""",
        default_factory=lambda: eval("Field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["dp"], "dp_field": []}


################################################################


class Type_perte_charge_dp_regul(Type_perte_charge_deriv):
    r"""
    Keyword used to regulate the DP value in order to match a target flow rate. Syntax :
    dp_regul { DP0 d deb d eps e }
    """

    dp0: float = Field(description=r"""initial value of DP""", default=0.0)
    deb: str = Field(description=r"""target flow rate in kg/s""", default="")
    eps: str = Field(
        description=r"""strength of the regulation (low values might be slow to find the target flow rate, high values might oscillate around the target value)""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: ["dp_regul"], "dp0": [], "deb": [], "eps": []}


################################################################


class Dp_impose(Source_base):
    r"""
    Source term to impose a pressure difference according to the formula : DP = dp + dDP/dQ *
    (Q - Q0)
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    dp_type: Type_perte_charge_deriv = Field(
        description=r"""mass flow rate (kg/s).""",
        default_factory=lambda: eval("Type_perte_charge_deriv()"),
    )
    surface: Literal["surface"] = Field(description=r"""not_set""", default="surface")
    bloc_surface: Bloc_lecture = Field(
        description=r"""Three syntaxes are possible for the surface definition block:  For VDF and VEF: { X|Y|Z = location subzone_name }  Only for VEF: { Surface surface_name }.  For polymac { Surface surface_name Orientation champ_uniforme }.""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: [],
        "aco": [],
        "dp_type": [],
        "surface": [],
        "bloc_surface": [],
        "acof": [],
    }


################################################################


class Acceleration(Source_base):
    r"""
    Momentum source term to take in account the forces due to rotation or translation of a non
    Galilean referential R\' (centre 0\') into the Galilean referential R (centre 0).
    """

    vitesse: Optional[Field_base] = Field(
        description=r"""Keyword for the velocity of the referential R\' into the R referential (dOO\'/dt term [m.s-1]). The velocity is mandatory when you want to print the total cinetic energy into the non-mobile Galilean referential R (see Ec_dans_repere_fixe keyword).""",
        default=None,
    )
    acceleration: Optional[Field_base] = Field(
        description=r"""Keyword for the acceleration of the referential R\' into the R referential (d2OO\'/dt2 term [m.s-2]). field_base is a time dependant field (eg: Champ_Fonc_t).""",
        default=None,
    )
    omega: Optional[Field_base] = Field(
        description=r"""Keyword for a rotation of the referential R\' into the R referential [rad.s-1]. field_base is a 3D time dependant field specified for example by a Champ_Fonc_t keyword. The time_field field should have 3 components even in 2D (In 2D: 0 0 omega).""",
        default=None,
    )
    domegadt: Optional[Field_base] = Field(
        description=r"""Keyword to define the time derivative of the previous rotation [rad.s-2]. Should be zero if the rotation is constant. The time_field field should have 3 components even in 2D (In 2D: 0 0 domegadt).""",
        default=None,
    )
    centre_rotation: Optional[Field_base] = Field(
        description=r"""Keyword to specify the centre of rotation (expressed in R\' coordinates) of R\' into R (if the domain rotates with the R\' referential, the centre of rotation is 0\'=(0,0,0)). The time_field should have 2 or 3 components according the dimension 2 or 3.""",
        default=None,
    )
    option: Optional[Literal["terme_complet", "coriolis_seul", "entrainement_seul"]] = (
        Field(
            description=r"""Keyword to specify the kind of calculation: terme_complet (default option) will calculate both the Coriolis and centrifugal forces, coriolis_seul will calculate the first one only, entrainement_seul will calculate the second one only.""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "vitesse": [],
        "acceleration": [],
        "omega": [],
        "domegadt": [],
        "centre_rotation": [],
        "option": [],
    }


################################################################


class Spec_pdcr_base(Objet_lecture):
    r"""
    Class to read the source term modelling the presence of a bundle of tubes in a flow. Cf=A
    Re-B.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Perte_charge_reguliere(Source_base):
    r"""
    Source term modelling the presence of a bundle of tubes in a flow.
    """

    spec: Spec_pdcr_base = Field(
        description=r"""Description of longitudinale or transversale type.""",
        default_factory=lambda: eval("Spec_pdcr_base()"),
    )
    zone_name: str = Field(
        description=r"""Name of the sub-area occupied by the tube bundle. A Sous_Zone (Sub-area) type object called zone_name should have been previously created.""",
        default="",
    )
    _synonyms: ClassVar[dict] = {None: [], "spec": [], "zone_name": ["name_of_zone"]}


################################################################


class Longitudinale(Spec_pdcr_base):
    r"""
    Class to define the pressure loss in the direction of the tube bundle.
    """

    dir: Literal["x", "y", "z"] = Field(description=r"""Direction.""", default="x")
    dd: float = Field(
        description=r"""Tube bundle hydraulic diameter value. This value is expressed in m.""",
        default=0.0,
    )
    ch_a: Literal["a", "cf"] = Field(
        description=r"""Keyword to be used to set law coefficient values for the coefficient of regular pressure losses.""",
        default="a",
    )
    a: float = Field(
        description=r"""Value of a law coefficient for regular pressure losses.""",
        default=0.0,
    )
    ch_b: Optional[Literal["b"]] = Field(
        description=r"""Keyword to be used to set law coefficient values for regular pressure losses.""",
        default=None,
    )
    b: Optional[float] = Field(
        description=r"""Value of a law coefficient for regular pressure losses.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dir": [],
        "dd": [],
        "ch_a": [],
        "a": [],
        "ch_b": [],
        "b": [],
    }


################################################################


class Transversale(Spec_pdcr_base):
    r"""
    Class to define the pressure loss in the direction perpendicular to the tube bundle.
    """

    dir: Literal["x", "y", "z"] = Field(description=r"""Direction.""", default="x")
    dd: float = Field(description=r"""Value of the tube bundle step.""", default=0.0)
    chaine_d: Literal["d"] = Field(
        description=r"""Keyword to be used to set the value of the tube external diameter.""",
        default="d",
    )
    d: float = Field(
        description=r"""Value of the tube external diameter.""", default=0.0
    )
    ch_a: Literal["a", "cf"] = Field(
        description=r"""Keyword to be used to set law coefficient values for the coefficient of regular pressure losses.""",
        default="a",
    )
    a: float = Field(
        description=r"""Value of a law coefficient for regular pressure losses.""",
        default=0.0,
    )
    ch_b: Optional[Literal["b"]] = Field(
        description=r"""Keyword to be used to set law coefficient values for regular pressure losses.""",
        default=None,
    )
    b: Optional[float] = Field(
        description=r"""Value of a law coefficient for regular pressure losses.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dir": [],
        "dd": [],
        "chaine_d": [],
        "d": [],
        "ch_a": [],
        "a": [],
        "ch_b": [],
        "b": [],
    }


################################################################


class Source_qdm(Source_base):
    r"""
    Momentum source term in the Navier-Stokes equations.
    """

    ch: Field_base = Field(
        description=r"""Field type.""", default_factory=lambda: eval("Field_base()")
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": ["champ"]}


################################################################


class Perte_charge_singuliere(Source_base):
    r"""
    Source term that is used to model a pressure loss over a surface area (transition through
    a grid, sudden enlargement) defined by the faces of elements located on the intersection
    of a subzone named subzone_name and a X,Y, or Z plane located at X,Y or Z = location.
    """

    dir: Literal["kx", "ky", "kz", "k"] = Field(
        description=r"""KX, KY or KZ designate directional pressure loss coefficients for respectively X, Y or Z direction. Or in the case where you chose a target flow rate with regul. Use K for isotropic pressure loss coefficient""",
        default="kx",
    )
    coeff: Optional[float] = Field(
        description=r"""Value (float) of friction coefficient (KX, KY, KZ).""",
        default=None,
    )
    regul: Optional[Bloc_lecture] = Field(
        description=r"""option to have adjustable K with flowrate target  { K0 valeur_initiale_de_k deb debit_cible eps intervalle_variation_mutiplicatif}.""",
        default=None,
    )
    surface: Bloc_lecture = Field(
        description=r"""Three syntaxes are possible for the surface definition block:  For VDF and VEF: { X|Y|Z = location subzone_name }  Only for VEF: { Surface surface_name }.  For polymac { Surface surface_name Orientation champ_uniforme }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dir": [],
        "coeff": [],
        "regul": [],
        "surface": [],
    }


################################################################


class Canal_perio(Source_base):
    r"""
    Momentum source term to maintain flow rate. The expression of the source term is:

    S(t) = (2*(Q(0) - Q(t))-(Q(0)-Q(t-dt))/(coeff*dt*area)


    Where:

    coeff=damping coefficient

    area=area of the periodic boundary

    Q(t)=flow rate at time t

    dt=time step


    Three files will be created during calculation on a datafile named DataFile.data. The
    first file contains the flow rate evolution. The second file is useful for resuming a
    calculation with the flow rate of the previous stopped calculation, and the last one
    contains the pressure gradient evolution:

    -DataFile_Channel_Flow_Rate_ProblemName_BoundaryName

    -DataFile_Channel_Flow_Rate_repr_ProblemName_BoundaryName

    -DataFile_Pressure_Gradient_ProblemName_BoundaryName
    """

    u_etoile: Optional[float] = Field(description=r"""not_set""", default=None)
    coeff: Optional[float] = Field(
        description=r"""Damping coefficient (optional, default value is 10).""",
        default=None,
    )
    h: Optional[float] = Field(
        description=r"""Half heigth of the channel.""", default=None
    )
    bord: str = Field(
        description=r"""The name of the (periodic) boundary normal to the flow direction.""",
        default="",
    )
    debit_impose: Optional[float] = Field(
        description=r"""Optional option to specify the aimed flow rate Q(0). If not used, Q(0) is computed by the code after the projection phase, where velocity initial conditions are slighlty changed to verify incompressibility.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "u_etoile": [],
        "coeff": [],
        "h": [],
        "bord": [],
        "debit_impose": [],
    }


################################################################


class Terme_puissance_thermique_echange_impose(Source_base):
    r"""
    Source term to impose thermal power according to formula : P = himp * (T - Text). Where T
    is the Trust temperature, Text is the outside temperature with which energy is exchanged
    via an exchange coefficient himp
    """

    himp: Field_base = Field(
        description=r"""the exchange coefficient""",
        default_factory=lambda: eval("Field_base()"),
    )
    text: Field_base = Field(
        description=r"""the outside temperature""",
        default_factory=lambda: eval("Field_base()"),
    )
    pid_controler_on_targer_power: Optional[Bloc_lecture] = Field(
        description=r"""PID_controler_on_targer_power bloc with parameters target_power (required), Kp, Ki and Kd (at least one of them should be provided)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "himp": [],
        "text": [],
        "pid_controler_on_targer_power": [],
    }


################################################################


class Coriolis(Source_base):
    r"""
    Keyword for a Coriolis term in hydraulic equation. Warning: Only available in VDF.
    """

    omega: str = Field(description=r"""Value of omega.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "omega": []}


################################################################


class Boussinesq_temperature(Source_base):
    r"""
    Class to describe a source term that couples the movement quantity equation and energy
    equation with the Boussinesq hypothesis.
    """

    t0: str = Field(
        description=r"""Reference temperature value (oC or K). It can also be a time dependant function since the 1.6.6 version.""",
        default="",
    )
    verif_boussinesq: Optional[int] = Field(
        description=r"""Keyword to check (1) or not (0) the reference value in comparison with the mean value in the domain. It is set to 1 by default.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "t0": [], "verif_boussinesq": []}


################################################################


class Boussinesq_concentration(Source_base):
    r"""
    Class to describe a source term that couples the movement quantity equation and
    constituent transport equation with the Boussinesq hypothesis.
    """

    c0: List[float] = Field(
        description=r"""Reference concentration field type. The only field type currently available is Champ_Uniforme (Uniform field).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "c0": []}


################################################################


class Frontiere_ouverte_pression_moyenne_imposee(Neumann):
    r"""
    Class for open boundary with pressure mean level imposed.
    """

    pext: float = Field(description=r"""Mean pressure.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "pext": []}


################################################################


class Frontiere_ouverte_concentration_imposee(Dirichlet):
    r"""
    Imposed concentration condition at an open boundary called bord (edge) (situation
    corresponding to a fluid inlet). This condition must be associated with an imposed inlet
    velocity condition.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_knudsen_non_negligeable(Dirichlet):
    r"""
    Boundary condition for number of Knudsen (Kn) above 0.001 where slip-flow condition
    appears: the velocity near the wall depends on the shear stress : Kn=l/L with l is the
    mean-free-path of the molecules and L a characteristic length scale.

    U(y=0)-Uwall=k(dU/dY)

    Where k is a coefficient given by several laws:

    Mawxell : k=(2-s)*l/s

    Bestok\&Karniadakis :k=(2-s)/s*L*Kn/(1+Kn)

    Xue\&Fan :k=(2-s)/s*L*tanh(Kn)

    s is a value between 0 and 2 named accomodation coefficient. s=1 seems a good value.

    Warning : The keyword is available for VDF calculation only for the moment.
    """

    name_champ_1: Literal["vitesse_paroi", "k"] = Field(
        description=r"""Field name.""", default="vitesse_paroi"
    )
    champ_1: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    name_champ_2: Literal["vitesse_paroi", "k"] = Field(
        description=r"""Field name.""", default="vitesse_paroi"
    )
    champ_2: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "name_champ_1": [],
        "champ_1": [],
        "name_champ_2": [],
        "champ_2": [],
    }


################################################################


class Paroi(Condlim_base):
    r"""
    Impermeability condition at a wall called bord (edge) (standard flux zero). This condition
    must be associated with a wall type hydraulic condition.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Temperature_imposee_paroi(Paroi_temperature_imposee):
    r"""
    Imposed temperature condition at the wall called bord (edge).
    """

    _synonyms: ClassVar[dict] = {None: ["enthalpie_imposee_paroi"], "ch": []}


################################################################


class Neumann_paroi_adiabatique(Neumann_homogene):
    r"""
    Adiabatic wall neumann boundary condition
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Frontiere_ouverte_pression_imposee(Neumann):
    r"""
    Imposed pressure condition at the open boundary called bord (edge). The imposed pressure
    field is expressed in Pa.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Champ_front_pression_from_u(Front_field_base):
    r"""
    this field is used to define a pressure field depending of a velocity field.
    """

    expression: str = Field(
        description=r"""value depending of a velocity (like $2*u_moy^2$).""", default=""
    )
    _synonyms: ClassVar[dict] = {None: [], "expression": []}


################################################################


class Champ_ostwald(Field_base):
    r"""
    This keyword is used to define the viscosity variation law:

    Mu(T)= K(T)*(D:D/2)**((n-1)/2)
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Chmoy_faceperio(Traitement_particulier_base):
    r"""
    non documente
    """

    bloc: Bloc_lecture = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Ec(Traitement_particulier_base):
    r"""
    Keyword to print total kinetic energy into the referential linked to the domain (keyword
    Ec). In the case where the domain is moving into a Galilean referential, the keyword
    Ec_dans_repere_fixe will print total kinetic energy in the Galilean referential whereas Ec
    will print the value calculated into the moving referential linked to the domain
    """

    ec: Optional[bool] = Field(description=r"""not_set""", default=None)
    ec_dans_repere_fixe: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    periode: Optional[float] = Field(
        description=r"""periode is the keyword to set the period of printing into the file datafile_Ec.son or datafile_Ec_dans_repere_fixe.son.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "ec": [],
        "ec_dans_repere_fixe": [],
        "periode": [],
    }


################################################################


class Canal(Traitement_particulier_base):
    r"""
    Keyword for statistics on a periodic plane channel.
    """

    dt_impr_moy_spat: Optional[float] = Field(
        description=r"""Period to print the spatial average (default value is 1e6).""",
        default=None,
    )
    dt_impr_moy_temp: Optional[float] = Field(
        description=r"""Period to print the temporal average (default value is 1e6).""",
        default=None,
    )
    debut_stat: Optional[float] = Field(
        description=r"""Time to start the temporal averaging (default value is 1e6).""",
        default=None,
    )
    fin_stat: Optional[float] = Field(
        description=r"""Time to end the temporal averaging (default value is 1e6).""",
        default=None,
    )
    pulsation_w: Optional[float] = Field(
        description=r"""Pulsation for phase averaging (in case of pulsating forcing term) (no default value).""",
        default=None,
    )
    nb_points_par_phase: Optional[int] = Field(
        description=r"""Number of samples to represent phase average all along a period (no default value).""",
        default=None,
    )
    reprise: Optional[str] = Field(
        description=r"""val_moy_temp_xxxxxx.sauv : Keyword to resume a calculation with previous averaged quantities.  Note that for thermal and turbulent problems, averages on temperature and turbulent viscosity are automatically calculated. To resume a calculation with phase averaging, val_moy_temp_xxxxxx.sauv_phase file is required on the directory where the job is submitted (this last file will be then automatically loaded by TRUST).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "dt_impr_moy_spat": [],
        "dt_impr_moy_temp": [],
        "debut_stat": [],
        "fin_stat": [],
        "pulsation_w": [],
        "nb_points_par_phase": [],
        "reprise": [],
    }


################################################################


class Temperature(Traitement_particulier_base):
    r"""
    not_set
    """

    bord: str = Field(description=r"""not_set""", default="")
    direction: int = Field(description=r"""not_set""", default=0)
    _synonyms: ClassVar[dict] = {None: [], "bord": [], "direction": []}


################################################################


class Pb_hydraulique(Pb_base):
    r"""
    Resolution of the Navier-Stokes equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_standard: Navier_stokes_standard = Field(
        description=r"""Navier-Stokes equations.""",
        default_factory=lambda: eval("Navier_stokes_standard()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_standard": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Listeqn(Listobj):
    r"""
    List of equations.
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Pb_avec_liste_conc(Pb_base):
    r"""
    Class to create a classical problem with a list of scalar concentration equations.
    """

    list_equations: Annotated[List[Eqn_base], "Listeqn"] = Field(
        description=r"""List of equations.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "list_equations": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_list_concentration(Pb_avec_liste_conc):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equation (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_temperature": [],
        "list_equations": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_list_concentration(Pb_avec_liste_conc):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "list_equations": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_cloned_concentration(Pb_base):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport equations (concentration diffusion convection).""",
            default=None,
        )
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equation (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "convection_diffusion_temperature": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_cloned_concentration(Pb_base):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport vectorial equation (concentration diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Fluide_reel_base(Fluide_base):
    r"""
    Class for real fluids.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Fluide_sodium_liquide(Fluide_reel_base):
    r"""
    Class for Fluide_sodium_liquide
    """

    p_ref: Optional[float] = Field(
        description=r"""Use to set the pressure value in the closure law. If not specified, the value of the pressure unknown will be used""",
        default=None,
    )
    t_ref: Optional[float] = Field(
        description=r"""Use to set the temperature value in the closure law. If not specified, the value of the temperature unknown will be used""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "p_ref": [],
        "t_ref": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Fluide_sodium_gaz(Fluide_reel_base):
    r"""
    Class for Fluide_sodium_liquide
    """

    p_ref: Optional[float] = Field(
        description=r"""Use to set the pressure value in the closure law. If not specified, the value of the pressure unknown will be used""",
        default=None,
    )
    t_ref: Optional[float] = Field(
        description=r"""Use to set the temperature value in the closure law. If not specified, the value of the temperature unknown will be used""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "p_ref": [],
        "t_ref": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Pb_thermohydraulique(Pb_base):
    r"""
    Resolution of thermohydraulic problem.
    """

    fluide_incompressible: Optional[Fluide_incompressible] = Field(
        description=r"""The fluid medium associated with the problem (only one possibility).""",
        default=None,
    )
    fluide_ostwald: Optional[Fluide_ostwald] = Field(
        description=r"""The fluid medium associated with the problem (only one possibility).""",
        default=None,
    )
    fluide_sodium_liquide: Optional[Fluide_sodium_liquide] = Field(
        description=r"""The fluid medium associated with the problem (only one possibility).""",
        default=None,
    )
    fluide_sodium_gaz: Optional[Fluide_sodium_gaz] = Field(
        description=r"""The fluid medium associated with the problem (only one possibility).""",
        default=None,
    )
    correlations: Optional[Bloc_lecture] = Field(
        description=r"""List of correlations used in specific source terms (i.e. interfacial flux, interfacial friction, ...)""",
        default=None,
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equation (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "fluide_ostwald": [],
        "fluide_sodium_liquide": [],
        "fluide_sodium_gaz": [],
        "correlations": [],
        "navier_stokes_standard": [],
        "convection_diffusion_temperature": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_avec_passif(Pb_base):
    r"""
    Class to create a classical problem with a scalar transport equation (e.g: temperature or
    concentration) and an additional set of passive scalars (e.g: temperature or
    concentration) equations.
    """

    equations_scalaires_passifs: Annotated[List[Eqn_base], "Listeqn"] = Field(
        description=r"""List of equations.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_concentration_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations, with the
    additional passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport equations (concentration diffusion convection).""",
            default=None,
        )
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equations (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "convection_diffusion_temperature": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of thermohydraulic problem, with the additional passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equations (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_temperature": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_concentration_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations with the additional
    passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport equations (concentration diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_concentration(Pb_base):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport vectorial equation (concentration diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_concentration(Pb_base):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_standard: Optional[Navier_stokes_standard] = Field(
        description=r"""Navier-Stokes equations.""", default=None
    )
    convection_diffusion_concentration: Optional[Convection_diffusion_concentration] = (
        Field(
            description=r"""Constituent transport equations (concentration diffusion convection).""",
            default=None,
        )
    )
    convection_diffusion_temperature: Optional[Convection_diffusion_temperature] = (
        Field(
            description=r"""Energy equation (temperature diffusion convection).""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_standard": [],
        "convection_diffusion_concentration": [],
        "convection_diffusion_temperature": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Negligeable(Turbulence_paroi_base):
    r"""
    Keyword to suppress the calculation of a law of the wall with a turbulence model. The wall
    stress is directly calculated with the derivative of the velocity, in the direction
    perpendicular to the wall (tau_tan /rho= nu dU/dy).

    Warning: This keyword is not available for k-epsilon models. In that case you must choose
    a wall law.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Negligeable_scalaire(Turbulence_paroi_scalaire_base):
    r"""
    Keyword to suppress the calculation of a law of the wall with a turbulence model for
    thermohydraulic problems. The wall stress is directly calculated with the derivative of
    the velocity, in the direction perpendicular to the wall.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Convection_diffusion_chaleur_qc(Eqn_base):
    r"""
    Temperature equation for a quasi-compressible fluid.
    """

    mode_calcul_convection: Optional[
        Literal["ancien", "divut_moins_tdivu", "divrhout_moins_tdivrhou"]
    ] = Field(
        description=r"""Option to set the form of the convective operator divrhouT_moins_Tdivrhou (the default since 1.6.8): rho.u.gradT = div(rho.u.T )- Tdiv(rho.u.1) ancien: u.gradT = div(u.T) - T.div(u)  divuT_moins_Tdivu : u.gradT = div(u.T) - Tdiv(u.1)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "mode_calcul_convection": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_chaleur_turbulent_qc(Convection_diffusion_chaleur_qc):
    r"""
    Temperature equation for a quasi-compressible fluid as well as the associated turbulence
    model equations.
    """

    modele_turbulence: Optional[Modele_turbulence_scal_base] = Field(
        description=r"""Turbulence model for the temperature (energy) conservation equation.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "mode_calcul_convection": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Navier_stokes_turbulent_qc(Navier_stokes_turbulent):
    r"""
    Navier-Stokes equations under low Mach number as well as the associated turbulence model
    equations.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Espece(Interprete):
    r"""
    not_set
    """

    mu: Field_base = Field(
        description=r"""Species dynamic viscosity value (kg.m-1.s-1).""",
        default_factory=lambda: eval("Field_base()"),
    )
    cp: Field_base = Field(
        description=r"""Species specific heat value (J.kg-1.K-1).""",
        default_factory=lambda: eval("Field_base()"),
    )
    masse_molaire: float = Field(description=r"""Species molar mass.""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["nul"], "mu": [], "cp": [], "masse_molaire": []}


################################################################


class Convection_diffusion_espece_multi_turbulent_qc(Eqn_base):
    r"""
    not_set
    """

    modele_turbulence: Optional[Modele_turbulence_scal_base] = Field(
        description=r"""Turbulence model to be used.""", default=None
    )
    espece: Espece = Field(
        description=r"""not_set""", default_factory=lambda: eval("Espece()")
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "espece": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_temperature_turbulent(Eqn_base):
    r"""
    Energy equation (temperature diffusion convection) as well as the associated turbulence
    model equations.
    """

    modele_turbulence: Optional[Modele_turbulence_scal_base] = Field(
        description=r"""Turbulence model for the energy equation.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_espece_binaire_qc(Eqn_base):
    r"""
    Species conservation equation for a binary quasi-compressible fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_espece_binaire_turbulent_qc(
    Convection_diffusion_espece_binaire_qc
):
    r"""
    Species conservation equation for a binary quasi-compressible fluid as well as the
    associated turbulence model equations.
    """

    modele_turbulence: Optional[Modele_turbulence_scal_base] = Field(
        description=r"""Turbulence model for the species conservation equation.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Paroi_decalee_robin(Condlim_base):
    r"""
    This keyword is used to designate a Robin boundary condition (a.u+b.du/dn=c) associated
    with the Pironneau methodology for the wall laws. The value of given by the delta option
    is the distance between the mesh (where symmetry boundary condition is applied) and the
    fictious wall. This boundary condition needs the definition of the dedicated source terms
    (Source_Robin or Source_Robin_Scalaire) according the equations used.
    """

    delta: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "delta": []}


################################################################


class Modele_turbulence_hyd_null(Modele_turbulence_hyd_deriv):
    r"""
    Null turbulence model (turbulent viscosity = 0) which can be used with a turbulent
    problem.
    """

    _synonyms: ClassVar[dict] = {
        None: ["null"],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Prandtl(Modele_turbulence_scal_base):
    r"""
    The Prandtl model. For the scalar equations, only the model based on Reynolds analogy is
    available. If K_Epsilon was selected in the hydraulic equation, Prandtl must be selected
    for the convection-diffusion temperature equation coupled to the hydraulic equation and
    Schmidt for the concentration equations.
    """

    prdt: Optional[str] = Field(
        description=r"""Keyword to modify the constant (Prdt) of Prandtl model : Alphat=Nut/Prdt Default value is 0.9""",
        default=None,
    )
    prandt_turbulent_fonction_nu_t_alpha: Optional[str] = Field(
        description=r"""Optional keyword to specify turbulent diffusivity (by default, alpha_t=nu_t/Prt) with another formulae, for example: alpha_t=nu_t2/(0,7*alpha+0,85*nu_tt) with the string nu_t*nu_t/(0,7*alpha+0,85*nu_t) where alpha is the thermal diffusivity.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "prdt": [],
        "prandt_turbulent_fonction_nu_t_alpha": [],
        "dt_impr_nusselt": [],
        "dt_impr_nusselt_mean_only": [],
        "turbulence_paroi": [],
    }


################################################################


class Schmidt(Modele_turbulence_scal_base):
    r"""
    The Schmidt model. For the scalar equations, only the model based on Reynolds analogy is
    available. If K_Epsilon was selected in the hydraulic equation, Schmidt must be selected
    for the convection-diffusion temperature equation coupled to the hydraulic equation and
    Schmidt for the concentration equations.
    """

    scturb: Optional[float] = Field(
        description=r"""Keyword to modify the constant (Sct) of Schmlidt model : Dt=Nut/Sct Default value is 0.7.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "scturb": [],
        "dt_impr_nusselt": [],
        "dt_impr_nusselt_mean_only": [],
        "turbulence_paroi": [],
    }


################################################################


class Modele_turbulence_scal_null(Modele_turbulence_scal_base):
    r"""
    Null scalar turbulence model (turbulent diffusivity = 0) which can be used with a
    turbulent problem.
    """

    turbulence_paroi: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    _synonyms: ClassVar[dict] = {
        None: ["null"],
        "dt_impr_nusselt": [],
        "dt_impr_nusselt_mean_only": [],
    }


################################################################


class Pb_thermohydraulique_turbulent_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of thermohydraulic problem, with turbulence modelling and with the additional
    passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_temperature_turbulent: Optional[
        Convection_diffusion_temperature_turbulent
    ] = Field(
        description=r"""Energy equations (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Fluide_dilatable_base(Fluide_base):
    r"""
    Basic class for dilatable fluids.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Bloc_sutherland(Objet_lecture):
    r"""
    Sutherland law for viscosity mu(T)=mu0*((T0+C)/(T+C))*(T/T0)**1.5 and (optional) for
    conductivity lambda(T)=mu0*Cp/Prandtl*((T0+Slambda)/(T+Slambda))*(T/T0)**1.5
    """

    problem_name: str = Field(description=r"""Name of problem.""", default="")
    mu0: Literal["mu0"] = Field(description=r"""not_set""", default="mu0")
    mu0_val: float = Field(description=r"""not_set""", default=0.0)
    t0: Literal["t0"] = Field(description=r"""not_set""", default="t0")
    t0_val: float = Field(description=r"""not_set""", default=0.0)
    slambda: Optional[Literal["slambda"]] = Field(
        description=r"""not_set""", default=None
    )
    s: Optional[float] = Field(description=r"""not_set""", default=None)
    c: Literal["c"] = Field(description=r"""not_set""", default="c")
    c_val: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "problem_name": [],
        "mu0": [],
        "mu0_val": [],
        "t0": [],
        "t0_val": [],
        "slambda": [],
        "s": [],
        "c": [],
        "c_val": [],
    }


################################################################


class Loi_etat_base(Objet_u):
    r"""
    Basic class for state laws used with a dilatable fluid.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Fluide_quasi_compressible(Fluide_dilatable_base):
    r"""
    Quasi-compressible flow with a low mach number assumption; this means that the thermo-
    dynamic pressure (used in state law) is uniform in space.
    """

    sutherland: Optional[Bloc_sutherland] = Field(
        description=r"""Sutherland law for viscosity and for conductivity.""",
        default=None,
    )
    pression: Optional[float] = Field(
        description=r"""Initial thermo-dynamic pressure used in the assosciated state law.""",
        default=None,
    )
    loi_etat: Optional[Loi_etat_base] = Field(
        description=r"""The state law that will be associated to the Quasi-compressible fluid.""",
        default=None,
    )
    traitement_pth: Optional[Literal["edo", "constant", "conservation_masse"]] = Field(
        description=r"""Particular treatment for the thermodynamic pressure Pth ; there are three possibilities:  1) with the keyword \'edo\' the code computes Pth solving an O.D.E. ; in this case, the mass is not strictly conserved (it is the default case for quasi compressible computation):  2) the keyword \'conservation_masse\' forces the conservation of the mass (closed geometry or with periodic boundaries condition)  3) the keyword \'constant\' makes it possible to have a constant Pth ; it\'s the good choice when the flow is open (e.g. with pressure boundary conditions).  It is possible to monitor the volume averaged value for temperature and density, plus Pth evolution in the .evol_glob file.""",
        default=None,
    )
    traitement_rho_gravite: Optional[Literal["standard", "moins_rho_moyen"]] = Field(
        description=r"""It may be :1) \`standard\` : the gravity term is evaluted with rho*g (It is the default). 2) \`moins_rho_moyen\` : the gravity term is evaluated with (rho-rhomoy) *g. Unknown pressure is then P*=P+rhomoy*g*z. It is useful when you apply uniforme pressure boundary condition like P*=0.""",
        default=None,
    )
    temps_debut_prise_en_compte_drho_dt: Optional[float] = Field(
        description=r"""While time<value, dRho/dt is set to zero (Rho, volumic mass). Useful for some calculation during the first time steps with big variation of temperature and volumic mass.""",
        default=None,
    )
    omega_relaxation_drho_dt: Optional[float] = Field(
        description=r"""Optional option to have a relaxed algorithm to solve the mass equation. value is used (1 per default) to specify omega.""",
        default=None,
    )
    lambda_: Optional[Field_base] = Field(
        description=r"""Conductivity (W.m-1.K-1).""", default=None
    )
    mu: Optional[Field_base] = Field(
        description=r"""Dynamic viscosity (kg.m-1.s-1).""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "sutherland": [],
        "pression": [],
        "loi_etat": [],
        "traitement_pth": [],
        "traitement_rho_gravite": [],
        "temps_debut_prise_en_compte_drho_dt": [],
        "omega_relaxation_drho_dt": [],
        "lambda_": ["lambda_u", "lambda"],
        "mu": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "cp": [],
    }


################################################################


class Pb_thermohydraulique_especes_turbulent_qc(Pb_avec_passif):
    r"""
    Resolution of turbulent thermohydraulic problem under low Mach number with passive scalar
    equations.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    navier_stokes_turbulent_qc: Navier_stokes_turbulent_qc = Field(
        description=r"""Navier-Stokes equations under low Mach number as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Navier_stokes_turbulent_qc()"),
    )
    convection_diffusion_chaleur_turbulent_qc: Convection_diffusion_chaleur_turbulent_qc = Field(
        description=r"""Energy equation under low Mach number as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_turbulent_qc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_turbulent_qc": [],
        "convection_diffusion_chaleur_turbulent_qc": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_concentration_turbulent_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations, with turbulence
    modelling and with the additional passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_concentration_turbulent_scalaires_passifs(Pb_avec_passif):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations, with
    turbulence modelling and with the additional passive scalar equations.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_temperature_turbulent: Optional[
        Convection_diffusion_temperature_turbulent
    ] = Field(
        description=r"""Energy equations (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_turbulent_qc(Pb_base):
    r"""
    Resolution of turbulent thermohydraulic problem under low Mach number.

    Warning : Available for VDF and VEF P0/P1NC discretization only.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    navier_stokes_turbulent_qc: Navier_stokes_turbulent_qc = Field(
        description=r"""Navier-Stokes equations under low Mach number as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Navier_stokes_turbulent_qc()"),
    )
    convection_diffusion_chaleur_turbulent_qc: Convection_diffusion_chaleur_turbulent_qc = Field(
        description=r"""Energy equation under low Mach number as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_turbulent_qc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_turbulent_qc": [],
        "convection_diffusion_chaleur_turbulent_qc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_list_concentration_turbulent(Pb_avec_liste_conc):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations, with
    turbulence modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_temperature_turbulent: Optional[
        Convection_diffusion_temperature_turbulent
    ] = Field(
        description=r"""Energy equation (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "list_equations": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_list_concentration_turbulent(Pb_avec_liste_conc):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations, with turbulence
    modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "list_equations": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_concentration_turbulent(Pb_base):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations, with turbulence
    modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_turbulent(Pb_base):
    r"""
    Resolution of thermohydraulic problem, with turbulence modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_turbulent: Navier_stokes_turbulent = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Navier_stokes_turbulent()"),
    )
    convection_diffusion_temperature_turbulent: Convection_diffusion_temperature_turbulent = Field(
        description=r"""Energy equation (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Convection_diffusion_temperature_turbulent()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_cloned_concentration_turbulent(Pb_base):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations, with
    turbulence modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_temperature_turbulent: Optional[
        Convection_diffusion_temperature_turbulent
    ] = Field(
        description=r"""Energy equation (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_cloned_concentration_turbulent(Pb_base):
    r"""
    Resolution of Navier-Stokes/multiple constituent transport equations, with turbulence
    modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_turbulent(Pb_base):
    r"""
    Resolution of Navier-Stokes equations with turbulence modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_turbulent: Navier_stokes_turbulent = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Navier_stokes_turbulent()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_concentration_turbulent(Pb_base):
    r"""
    Resolution of Navier-Stokes/energy/multiple constituent transport equations, with
    turbulence modelling.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_turbulent: Optional[Navier_stokes_turbulent] = Field(
        description=r"""Navier-Stokes equations as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_concentration_turbulent: Optional[
        Convection_diffusion_concentration_turbulent
    ] = Field(
        description=r"""Constituent transport equations (concentration diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    convection_diffusion_temperature_turbulent: Optional[
        Convection_diffusion_temperature_turbulent
    ] = Field(
        description=r"""Energy equation (temperature diffusion convection) as well as the associated turbulence model equations.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_concentration_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_melange_binaire_turbulent_qc(Pb_base):
    r"""
    Resolution of a turbulent binary mixture problem for a quasi-compressible fluid with an
    iso-thermal condition.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    navier_stokes_turbulent_qc: Navier_stokes_turbulent_qc = Field(
        description=r"""Navier-Stokes equation for a quasi-compressible fluid as well as the associated turbulence model equations.""",
        default_factory=lambda: eval("Navier_stokes_turbulent_qc()"),
    )
    convection_diffusion_espece_binaire_turbulent_qc: Convection_diffusion_espece_binaire_turbulent_qc = Field(
        description=r"""Species conservation equation for a quasi-compressible fluid as well as the associated turbulence model equations.""",
        default_factory=lambda: eval(
            "Convection_diffusion_espece_binaire_turbulent_qc()"
        ),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_turbulent_qc": [],
        "convection_diffusion_espece_binaire_turbulent_qc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Reactions(Listobj):
    r"""
    list of reactions
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Reaction(Objet_lecture):
    r"""
    Keyword to describe reaction:

    w =K pow(T,beta) exp(-Ea/( R T)) $\Pi$ pow(Reactif_i,activitivity_i).

    If K_inv >0,

    w= K pow(T,beta) exp(-Ea/( R T)) ( $\Pi$ pow(Reactif_i,activitivity_i) -
    Kinv/exp(-c_r_Ea/(R T)) $\Pi$ pow(Produit_i,activitivity_i ))
    """

    reactifs: str = Field(description=r"""LHS of equation (ex CH4+2*O2)""", default="")
    produits: str = Field(description=r"""RHS of equation (ex CO2+2*H20)""", default="")
    constante_taux_reaction: Optional[float] = Field(
        description=r"""constante of cinetic K""", default=None
    )
    enthalpie_reaction: float = Field(description=r"""DH""", default=0.0)
    energie_activation: float = Field(description=r"""Ea""", default=0.0)
    exposant_beta: float = Field(description=r"""Beta""", default=0.0)
    coefficients_activites: Optional[Bloc_lecture] = Field(
        description=r"""coefficients od ativity (exemple { CH4 1 O2 2 })""",
        default=None,
    )
    contre_reaction: Optional[float] = Field(description=r"""K_inv""", default=None)
    contre_energie_activation: Optional[float] = Field(
        description=r"""c_r_Ea""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "reactifs": [],
        "produits": [],
        "constante_taux_reaction": [],
        "enthalpie_reaction": [],
        "energie_activation": [],
        "exposant_beta": [],
        "coefficients_activites": [],
        "contre_reaction": [],
        "contre_energie_activation": [],
    }


################################################################


class Chimie(Objet_u):
    r"""
    Keyword to describe the chmical reactions
    """

    reactions: Annotated[List[Reaction], "Reactions"] = Field(
        description=r"""list of reactions""", default_factory=list
    )
    modele_micro_melange: Optional[int] = Field(
        description=r"""modele_micro_melange (0 by default)""", default=None
    )
    constante_modele_micro_melange: Optional[float] = Field(
        description=r"""constante of modele (1 by default)""", default=None
    )
    espece_en_competition_micro_melange: Optional[str] = Field(
        description=r"""espece in competition in reactions""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "reactions": [],
        "modele_micro_melange": [],
        "constante_modele_micro_melange": [],
        "espece_en_competition_micro_melange": [],
    }


################################################################


class Type_diffusion_turbulente_multiphase_sgdh(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    not_set
    """

    pr_t: Optional[float] = Field(description=r"""not_set""", default=None)
    sigma: Optional[float] = Field(description=r"""not_set""", default=None)
    no_alpha: Optional[bool] = Field(description=r"""not_set""", default=None)
    gas_turb: Optional[bool] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["sgdh"],
        "pr_t": ["prandtl_turbulent"],
        "sigma": ["sigma_turbulent"],
        "no_alpha": [],
        "gas_turb": [],
    }


################################################################


class Type_diffusion_turbulente_multiphase_smago(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    LES Smagorinsky type.
    """

    cs: Optional[float] = Field(
        description=r"""Smagorinsky's model constant. By default it is se to 0.18.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["smago"], "cs": []}


################################################################


class Type_diffusion_turbulente_multiphase_wale(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    LES WALE type.
    """

    cw: Optional[float] = Field(
        description=r"""WALE's model constant. By default it is se to 0.5.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["wale"], "cw": []}


################################################################


class Diffusion_turbulente_multiphase(Diffusion_deriv):
    r"""
    Turbulent diffusion operator for multiphase problem
    """

    type: Optional[Type_diffusion_turbulente_multiphase_deriv] = Field(
        description=r"""Turbulence model for multiphase problem""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["turbulente"], "type": []}


################################################################


class Type_diffusion_turbulente_multiphase_prandtl(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    Scalar Prandtl model.
    """

    pr_t: Optional[float] = Field(
        description=r"""Prandtl's model constant. By default it is se to 0.9.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: ["prandtl"], "pr_t": ["prandtl_turbulent"]}


################################################################


class Type_diffusion_turbulente_multiphase_l_melange(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    not_set
    """

    l_melange: float = Field(description=r"""not_set""", default=0.0)
    _synonyms: ClassVar[dict] = {None: ["l_melange"], "l_melange": []}


################################################################


class Interface_base(Objet_u):
    r"""
    Basic class for a liquid-gas interface (used in pb_multiphase)
    """

    tension_superficielle: Optional[float] = Field(
        description=r"""surface tension""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "tension_superficielle": ["surface_tension"]}


################################################################


class Saturation_base(Interface_base):
    r"""
    fluide-gas interface with phase change (used in pb_multiphase)
    """

    p_ref: Optional[float] = Field(description=r"""not_set""", default=None)
    t_ref: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "p_ref": [],
        "t_ref": [],
        "tension_superficielle": ["surface_tension"],
    }


################################################################


class Saturation_sodium(Saturation_base):
    r"""
    Class for saturation sodium
    """

    p_ref: Optional[float] = Field(
        description=r"""Use to fix the pressure value in the closure law. If not specified, the value of the pressure unknown will be used""",
        default=None,
    )
    t_ref: Optional[float] = Field(
        description=r"""Use to fix the temperature value in the closure law. If not specified, the value of the temperature unknown will be used""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "p_ref": [],
        "t_ref": [],
        "tension_superficielle": ["surface_tension"],
    }


################################################################


class Milieu_musig(Listobj):
    r"""
    MUSIG medium made of several sub mediums.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Interface_sigma_constant(Interface_base):
    r"""
    Liquid-gas interface with a constant surface tension sigma
    """

    _synonyms: ClassVar[dict] = {None: [], "tension_superficielle": ["surface_tension"]}


################################################################


class Fluide_stiffened_gas(Fluide_reel_base):
    r"""
    Class for Stiffened Gas
    """

    gamma: Optional[float] = Field(
        description=r"""Heat capacity ratio (Cp/Cv)""", default=None
    )
    pinf: Optional[float] = Field(
        description=r"""Stiffened gas pressure constant (if set to zero, the state law becomes identical to that of perfect gases)""",
        default=None,
    )
    mu: Optional[float] = Field(description=r"""Dynamic viscosity""", default=None)
    lambda_: Optional[float] = Field(
        description=r"""Thermal conductivity""", default=None
    )
    cv: Optional[float] = Field(
        description=r"""Thermal capacity at constant volume""", default=None
    )
    q: Optional[float] = Field(description=r"""Reference energy""", default=None)
    q_prim: Optional[float] = Field(description=r"""Model constant""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "gamma": [],
        "pinf": [],
        "mu": [],
        "lambda_": ["lambda_u", "lambda"],
        "cv": [],
        "q": [],
        "q_prim": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "cp": [],
    }


################################################################


class Milieu_composite(Listobj):
    r"""
    Composite medium made of several sub mediums.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Saturation_constant(Saturation_base):
    r"""
    Class for saturation constant
    """

    p_sat: Optional[float] = Field(
        description=r"""Define the saturation pressure value (this is a required parameter)""",
        default=None,
    )
    t_sat: Optional[float] = Field(
        description=r"""Define the saturation temperature value (this is a required parameter)""",
        default=None,
    )
    lvap: Optional[float] = Field(
        description=r"""Latent heat of vaporization""", default=None
    )
    hlsat: Optional[float] = Field(
        description=r"""Liquid saturation enthalpy""", default=None
    )
    hvsat: Optional[float] = Field(
        description=r"""Vapor saturation enthalpy""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "p_sat": [],
        "t_sat": [],
        "lvap": [],
        "hlsat": [],
        "hvsat": [],
        "p_ref": [],
        "t_ref": [],
        "tension_superficielle": ["surface_tension"],
    }


################################################################


class Energie_multiphase_enthalpie(Eqn_base):
    r"""
    Internal energy conservation equation for a multi-phase problem where the unknown is the
    enthalpy
    """

    _synonyms: ClassVar[dict] = {
        None: ["energie_multiphase_h"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Qdm_multiphase(Eqn_base):
    r"""
    Momentum conservation equation for a multi-phase problem where the unknown is the velocity
    """

    solveur_pression: Optional[Solveur_sys_base] = Field(
        description=r"""Linear pressure system resolution method.""", default=None
    )
    evanescence: Optional[Bloc_lecture] = Field(
        description=r"""Management of the vanishing phase (when alpha tends to 0 or 1)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "solveur_pression": [],
        "evanescence": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Energie_multiphase(Eqn_base):
    r"""
    Internal energy conservation equation for a multi-phase problem where the unknown is the
    temperature
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Masse_multiphase(Eqn_base):
    r"""
    Mass consevation equation for a multi-phase problem where the unknown is the alpha (void
    fraction)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Vitesse_relative_base(Source_base):
    r"""
    Basic class for drift-velocity source term between a liquid and a gas phase
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Vitesse_derive_base(Vitesse_relative_base):
    r"""
    Source term which corresponds to the drift-velocity between a liquid and a gas phase
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Frottement_interfacial(Source_base):
    r"""
    Source term which corresponds to the phases friction at the interface
    """

    a_res: Optional[float] = Field(
        description=r"""void fraction at which the gas velocity is forced to approach liquid velocity (default alpha_evanescence*100)""",
        default=None,
    )
    dv_min: Optional[float] = Field(
        description=r"""minimal relative velocity used to linearize interfacial friction at low velocities""",
        default=None,
    )
    exp_res: Optional[int] = Field(
        description=r"""exponent that callibrates intensity of velocity convergence (default 2)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "a_res": [], "dv_min": [], "exp_res": []}


################################################################


class Flux_interfacial(Source_base):
    r"""
    Source term of mass transfer between phases connected by the saturation object defined in
    saturation_xxxx
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Portance_interfaciale(Source_base):
    r"""
    Base class for source term of lift force in momentum equation.
    """

    beta: Optional[float] = Field(
        description=r"""Multiplying factor for the bubble lift force source term.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "beta": []}


################################################################


class Travail_pression(Source_base):
    r"""
    Source term which corresponds to the additional pressure work term that appears when
    dealing with compressible multiphase fluids
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Dispersion_bulles(Source_base):
    r"""
    Base class for source terms of bubble dispersion in momentum equation.
    """

    beta: Optional[float] = Field(
        description=r"""Mutliplying factor for the output of the bubble dispersion source term.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {None: [], "beta": []}


################################################################


class Echelle_temporelle_turbulente(Eqn_base):
    r"""
    Turbulent Dissipation time scale equation for a turbulent mono/multi-phase problem
    (available in TrioCFD)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Energie_cinetique_turbulente(Eqn_base):
    r"""
    Turbulent kinetic Energy conservation equation for a turbulent mono/multi-phase problem
    (available in TrioCFD)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Energie_cinetique_turbulente_wit(Eqn_base):
    r"""
    Bubble Induced Turbulent kinetic Energy equation for a turbulent multi-phase problem
    (available in TrioCFD)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Taux_dissipation_turbulent(Eqn_base):
    r"""
    Turbulent Dissipation frequency equation for a turbulent mono/multi-phase problem
    (available in TrioCFD)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_multiphase(Pb_base):
    r"""
    A problem that allows the resolution of N-phases with 3*N equations
    """

    milieu_composite: Optional[Bloc_lecture] = Field(
        description=r"""The composite medium associated with the problem.""",
        default=None,
    )
    correlations: Optional[Bloc_lecture] = Field(
        description=r"""List of correlations used in specific source terms (i.e. interfacial flux, interfacial friction, ...)""",
        default=None,
    )
    models: Optional[Bloc_lecture] = Field(
        description=r"""List of models used in specific source terms (i.e. interfacial flux, interfacial friction, ...)""",
        default=None,
    )
    milieu_musig: Optional[Bloc_lecture] = Field(
        description=r"""The composite medium associated with the problem.""",
        default=None,
    )
    qdm_multiphase: Qdm_multiphase = Field(
        description=r"""Momentum conservation equation for a multi-phase problem where the unknown is the velocity""",
        default_factory=lambda: eval("Qdm_multiphase()"),
    )
    masse_multiphase: Masse_multiphase = Field(
        description=r"""Mass consevation equation for a multi-phase problem where the unknown is the alpha (void fraction)""",
        default_factory=lambda: eval("Masse_multiphase()"),
    )
    energie_multiphase: Energie_multiphase = Field(
        description=r"""Internal energy conservation equation for a multi-phase problem where the unknown is the temperature""",
        default_factory=lambda: eval("Energie_multiphase()"),
    )
    echelle_temporelle_turbulente: Optional[Echelle_temporelle_turbulente] = Field(
        description=r"""Turbulent Dissipation time scale equation for a turbulent mono/multi-phase problem (available in TrioCFD)""",
        default=None,
    )
    energie_cinetique_turbulente: Optional[Energie_cinetique_turbulente] = Field(
        description=r"""Turbulent kinetic Energy conservation equation for a turbulent mono/multi-phase problem (available in TrioCFD)""",
        default=None,
    )
    energie_cinetique_turbulente_wit: Optional[Energie_cinetique_turbulente_wit] = (
        Field(
            description=r"""Bubble Induced Turbulent kinetic Energy equation for a turbulent multi-phase problem (available in TrioCFD)""",
            default=None,
        )
    )
    taux_dissipation_turbulent: Optional[Taux_dissipation_turbulent] = Field(
        description=r"""Turbulent Dissipation frequency equation for a turbulent mono/multi-phase problem (available in TrioCFD)""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "milieu_composite": [],
        "correlations": [],
        "models": [],
        "milieu_musig": [],
        "qdm_multiphase": [],
        "masse_multiphase": [],
        "energie_multiphase": [],
        "echelle_temporelle_turbulente": [],
        "energie_cinetique_turbulente": [],
        "energie_cinetique_turbulente_wit": [],
        "taux_dissipation_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_multiphase_enthalpie(Pb_multiphase):
    r"""
    A problem that allows the resolution of N-phases with 3*N equations
    """

    milieu_composite: Optional[Bloc_lecture] = Field(
        description=r"""The composite medium associated with the problem.""",
        default=None,
    )
    correlations: Optional[Bloc_lecture] = Field(
        description=r"""List of correlations used in specific source terms (i.e. interfacial flux, interfacial friction, ...)""",
        default=None,
    )
    qdm_multiphase: Qdm_multiphase = Field(
        description=r"""Momentum conservation equation for a multi-phase problem where the unknown is the velocity""",
        default_factory=lambda: eval("Qdm_multiphase()"),
    )
    masse_multiphase: Masse_multiphase = Field(
        description=r"""Mass consevation equation for a multi-phase problem where the unknown is the alpha (void fraction)""",
        default_factory=lambda: eval("Masse_multiphase()"),
    )
    energie_multiphase_h: Energie_multiphase_enthalpie = Field(
        description=r"""Internal energy conservation equation for a multi-phase problem where the unknown is the enthalpy""",
        default_factory=lambda: eval("Energie_multiphase_enthalpie()"),
    )
    energie_multiphase: ClassVar[str] = Field(
        description=r"""suppress_param""", default="suppress_param"
    )
    _synonyms: ClassVar[dict] = {
        None: ["pb_multiphase_h"],
        "milieu_composite": [],
        "correlations": [],
        "qdm_multiphase": [],
        "masse_multiphase": [],
        "energie_multiphase_h": ["energie_multiphase_enthalpie"],
        "models": [],
        "milieu_musig": [],
        "echelle_temporelle_turbulente": [],
        "energie_cinetique_turbulente": [],
        "energie_cinetique_turbulente_wit": [],
        "taux_dissipation_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_multiphase_hem(Pb_multiphase):
    r"""
    A problem that allows the resolution of 2-phases mechanicaly and thermally coupled with 3
    equations
    """

    _synonyms: ClassVar[dict] = {
        None: ["pb_hem"],
        "milieu_composite": [],
        "correlations": [],
        "models": [],
        "milieu_musig": [],
        "qdm_multiphase": [],
        "masse_multiphase": [],
        "energie_multiphase": [],
        "echelle_temporelle_turbulente": [],
        "energie_cinetique_turbulente": [],
        "energie_cinetique_turbulente_wit": [],
        "taux_dissipation_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Simpler(Solveur_implicite_base):
    r"""
    Simpler method for incompressible systems.
    """

    seuil_convergence_implicite: float = Field(
        description=r"""Keyword to set the value of the convergence criteria for the resolution of the implicit system build to solve either the Navier_Stokes equation (only for Simple and Simpler algorithms) or a scalar equation. It is adviced to use the default value (1e6) to solve the implicit system only once by time step. This value must be decreased when a coupling between problems is considered.""",
        default=0.0,
    )
    seuil_convergence_solveur: Optional[float] = Field(
        description=r"""value of the convergence criteria for the resolution of the implicit system build by solving several times per time step the Navier_Stokes equation and the scalar equations if any. This value MUST be used when a coupling between problems is considered (should be set to a value typically of 0.1 or 0.01).""",
        default=None,
    )
    seuil_generation_solveur: Optional[float] = Field(
        description=r"""Option to create a GMRES solver and use vrel as the convergence threshold (implicit linear system Ax=B will be solved if residual error ||Ax-B|| is lesser than vrel).""",
        default=None,
    )
    seuil_verification_solveur: Optional[float] = Field(
        description=r"""Option to check if residual error ||Ax-B|| is lesser than vrel after the implicit linear system Ax=B has been solved.""",
        default=None,
    )
    seuil_test_preliminaire_solveur: Optional[float] = Field(
        description=r"""Option to decide if the implicit linear system Ax=B should be solved by checking if the residual error ||Ax-B|| is bigger than vrel.""",
        default=None,
    )
    solveur: Optional[Solveur_sys_base] = Field(
        description=r"""Method (different from the default one, Gmres with diagonal preconditioning) to solve the linear system.""",
        default=None,
    )
    no_qdm: Optional[bool] = Field(
        description=r"""Keyword to not solve qdm equation (and turbulence models of these equation).""",
        default=None,
    )
    nb_it_max: Optional[int] = Field(
        description=r"""Keyword to set the maximum iterations number for the Gmres.""",
        default=None,
    )
    controle_residu: Optional[bool] = Field(
        description=r"""Keyword of Boolean type (by default 0). If set to 1, the convergence occurs if the residu suddenly increases.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Bloc_criteres_convergence(Bloc_lecture):
    r"""
    Not set
    """

    _synonyms: ClassVar[dict] = {None: ["nul"], "bloc_lecture": []}


################################################################


class Sets(Simpler):
    r"""
    Stability-Enhancing Two-Step solver which is useful for a multiphase problem. Ref : J. H.
    MAHAFFY, A stability-enhancing two-step method for fluid flow calculations, Journal of
    Computational Physics, 46, 3, 329 (1982).
    """

    criteres_convergence: Optional[Bloc_criteres_convergence] = Field(
        description=r"""Set the convergence thresholds for each unknown (i.e: alpha, temperature, velocity and pressure). The default values are respectively 0.01, 0.1, 0.01 and 100""",
        default=None,
    )
    iter_min: Optional[int] = Field(
        description=r"""Number of minimum iterations (default value 1)""", default=None
    )
    iter_max: Optional[int] = Field(
        description=r"""Number of maximum iterations (default value 10)""", default=None
    )
    seuil_convergence_implicite: Optional[float] = Field(
        description=r"""Convergence criteria.""", default=None
    )
    nb_corrections_max: Optional[int] = Field(
        description=r"""Maximum number of corrections performed by the PISO algorithm to achieve the projection of the velocity field. The algorithm may perform less corrections then nb_corrections_max if the accuracy of the projection is sufficient. (By default nb_corrections_max is set to 21).""",
        default=None,
    )
    facsec_diffusion_for_sets: Optional[float] = Field(
        description=r"""facsec to impose on the diffusion time step in sets while the total time step stays smaller than the convection time step.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "criteres_convergence": [],
        "iter_min": [],
        "iter_max": [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "facsec_diffusion_for_sets": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Ice(Sets):
    r"""
    Implicit Continuous-fluid Eulerian solver which is useful for a multiphase problem. Robust
    pressure reduction resolution.
    """

    pression_degeneree: Optional[int] = Field(
        description=r"""Set to 1 if the pressure field is degenerate (ex. : incompressible fluid with no imposed-pressure BCs). Default: autodetected""",
        default=None,
    )
    reduction_pression: Optional[int] = Field(
        description=r"""Set to 1 if the user wants a resolution with a pressure reduction. Otherwise, the flag is to be set to 0 so that the complete matrix is considered. The default value of this flag is 1.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "pression_degeneree": [],
        "reduction_pression": ["pressure_reduction"],
        "criteres_convergence": [],
        "iter_min": [],
        "iter_max": [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "facsec_diffusion_for_sets": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Criteres_convergence(Interprete):
    r"""
    convergence criteria
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    inco: Optional[str] = Field(
        description=r"""Unknown (i.e: alpha, temperature, velocity and pressure)""",
        default=None,
    )
    val: Optional[float] = Field(description=r"""Convergence threshold""", default=None)
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "aco": [],
        "inco": [],
        "val": [],
        "acof": [],
    }


################################################################


class Piso(Simpler):
    r"""
    Piso (Pressure Implicit with Split Operator) - method to solve N_S.
    """

    seuil_convergence_implicite: Optional[float] = Field(
        description=r"""Convergence criteria.""", default=None
    )
    nb_corrections_max: Optional[int] = Field(
        description=r"""Maximum number of corrections performed by the PISO algorithm to achieve the projection of the velocity field. The algorithm may perform less corrections then nb_corrections_max if the accuracy of the projection is sufficient. (By default nb_corrections_max is set to 21).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Implicite(Piso):
    r"""
    similar to PISO, but as it looks like a simplified solver, it will use fewer timesteps.
    But it may run faster because the pressure matrix is not re-assembled and thus provides
    CPU gains.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Simple(Piso):
    r"""
    SIMPLE type algorithm
    """

    relax_pression: Optional[float] = Field(
        description=r"""Value between 0 and 1 (by default 1), this keyword is used only by the SIMPLE algorithm for relaxing the increment of pressure.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "relax_pression": [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Solveur_u_p(Simple):
    r"""
    similar to simple.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "relax_pression": [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Sch_cn_iteratif(Schema_temps_base):
    r"""
    The Crank-Nicholson method of second order accuracy. A mid-point rule formulation is used
    (Euler-centered scheme). The basic scheme is: $$u(t+1) = u(t) + du/dt(t+1/2)*dt$$ The
    estimation of the time derivative du/dt at the level (t+1/2) is obtained either by
    iterative process. The time derivative du/dt at the level (t+1/2) is calculated
    iteratively with a simple under-relaxations method. Since the method is implicit, neither
    the cfl nor the fourier stability criteria must be respected. The time step is calculated
    in a way that the iterative procedure converges with the less iterations as possible.

    Remark : for stationary or RANS calculations, no limitation can be given for time step
    through high value of facsec_max parameter (for instance : facsec_max 1000). In
    counterpart, for LES calculations, high values of facsec_max may engender numerical
    instabilities.
    """

    seuil: Optional[float] = Field(
        description=r"""criteria for ending iterative process (Max( || u(p) - u(p-1)||/Max || u(p) ||) < seuil) (0.001 by default)""",
        default=None,
    )
    niter_min: Optional[int] = Field(
        description=r"""minimal number of p-iterations to satisfy convergence criteria (2 by default)""",
        default=None,
    )
    niter_max: Optional[int] = Field(
        description=r"""number of maximum p-iterations allowed to satisfy convergence criteria (6 by default)""",
        default=None,
    )
    niter_avg: Optional[int] = Field(
        description=r"""threshold of p-iterations (3 by default). If the number of p-iterations is greater than niter_avg, facsec is reduced, if lesser than niter_avg, facsec is increased (but limited by the facsec_max value).""",
        default=None,
    )
    facsec_max: Optional[float] = Field(
        description=r"""maximum ratio allowed between dynamical time step returned by iterative process and stability time returned by CFL condition (2 by default).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil": [],
        "niter_min": [],
        "niter_max": [],
        "niter_avg": [],
        "facsec_max": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_backward_differentiation_order_2(Schema_implicite_base):
    r"""
    not_set
    """

    facsec_max: Optional[float] = Field(
        description=r"""Maximum ratio allowed between time step and stability time returned by CFL condition. The initial ratio given by facsec keyword is changed during the calculation with the implicit scheme but it couldn\'t be higher than facsec_max value. Warning: Some implicit schemes do not permit high facsec_max, example Schema_Adams_Moulton_order_3 needs facsec=facsec_max=1.  Advice: The calculation may start with a facsec specified by the user and increased by the algorithm up to the facsec_max limit. But the user can also choose to specify a constant facsec (facsec_max will be set to facsec value then). Faster convergence has been seen and depends on the kind of calculation: -Hydraulic only or thermal hydraulic with forced convection and low coupling between velocity and temperature (Boussinesq value beta low), facsec between 20-30-Thermal hydraulic with forced convection and strong coupling between velocity and temperature (Boussinesq value beta high), facsec between 90-100 -Thermohydralic with natural convection, facsec around 300 -Conduction only, facsec can be set to a very high value (1e8) as if the scheme was unconditionally stableThese values can also be used as rule of thumb for initial facsec with a facsec_max limit higher.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "facsec_max": [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Schema_backward_differentiation_order_3(Schema_implicite_base):
    r"""
    not_set
    """

    facsec_max: Optional[float] = Field(
        description=r"""Maximum ratio allowed between time step and stability time returned by CFL condition. The initial ratio given by facsec keyword is changed during the calculation with the implicit scheme but it couldn\'t be higher than facsec_max value. Warning: Some implicit schemes do not permit high facsec_max, example Schema_Adams_Moulton_order_3 needs facsec=facsec_max=1.  Advice: The calculation may start with a facsec specified by the user and increased by the algorithm up to the facsec_max limit. But the user can also choose to specify a constant facsec (facsec_max will be set to facsec value then). Faster convergence has been seen and depends on the kind of calculation: -Hydraulic only or thermal hydraulic with forced convection and low coupling between velocity and temperature (Boussinesq value beta low), facsec between 20-30-Thermal hydraulic with forced convection and strong coupling between velocity and temperature (Boussinesq value beta high), facsec between 90-100 -Thermohydralic with natural convection, facsec around 300 -Conduction only, facsec can be set to a very high value (1e8) as if the scheme was unconditionally stableThese values can also be used as rule of thumb for initial facsec with a facsec_max limit higher.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "facsec_max": [],
        "max_iter_implicite": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Sch_cn_ex_iteratif(Sch_cn_iteratif):
    r"""
    This keyword also describes a Crank-Nicholson method of second order accuracy but here,
    for scalars, because of instablities encountered when dt>dt_CFL, the Crank Nicholson
    scheme is not applied to scalar quantities. Scalars are treated according to Euler-
    Explicite scheme at the end of the CN treatment for velocity flow fields (by doing p Euler
    explicite under-iterations at dt<=dt_CFL). Parameters are the sames (but default values
    may change) compare to the Sch_CN_iterative scheme plus a relaxation keyword: niter_min (2
    by default), niter_max (6 by default), niter_avg (3 by default), facsec_max (20 by
    default), seuil (0.05 by default)
    """

    omega: Optional[float] = Field(
        description=r"""relaxation factor (0.1 by default)""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "omega": [],
        "seuil": [],
        "niter_min": [],
        "niter_max": [],
        "niter_avg": [],
        "facsec_max": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Loi_etat_tppi_base(Loi_etat_base):
    r"""
    Basic class for thermo-physical properties interface (TPPI) used for dilatable problems
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Coolprop_qc(Loi_etat_tppi_base):
    r"""
    Class for using CoolProp with QC problem
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    fluid: str = Field(description=r"""Fluid name in the CoolProp model""", default="")
    model: str = Field(description=r"""CoolProp model name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "cp": [], "fluid": [], "model": []}


################################################################


class Loi_etat_gaz_reel_base(Loi_etat_base):
    r"""
    Basic class for real gases state laws used with a dilatable fluid.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Rhot_gaz_reel_qc(Loi_etat_gaz_reel_base):
    r"""
    Class for real gas state law used with a quasi-compressible fluid.
    """

    bloc: Bloc_lecture = Field(
        description=r"""Description.""", default_factory=lambda: eval("Bloc_lecture()")
    )
    _synonyms: ClassVar[dict] = {None: [], "bloc": []}


################################################################


class Loi_etat_gaz_parfait_base(Loi_etat_base):
    r"""
    Basic class for perfect gases state laws used with a dilatable fluid.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Multi_gaz_parfait_qc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas multi-species mixtures state law used with a quasi-compressible
    fluid.
    """

    sc: float = Field(
        description=r"""Schmidt number of the gas Sc=nu/D (D: diffusion coefficient of the mixing).""",
        default=0.0,
    )
    prandtl: float = Field(
        description=r"""Prandtl number of the gas Pr=mu*Cp/lambda""", default=0.0
    )
    cp: Optional[float] = Field(
        description=r"""Specific heat at constant pressure of the gas Cp.""",
        default=None,
    )
    dtol_fraction: Optional[float] = Field(
        description=r"""Delta tolerance on mass fractions for check testing (default value 1.e-6).""",
        default=None,
    )
    correction_fraction: Optional[bool] = Field(
        description=r"""To force mass fractions between 0. and 1.""", default=None
    )
    ignore_check_fraction: Optional[bool] = Field(
        description=r"""Not to check if mass fractions between 0. and 1.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "sc": [],
        "prandtl": [],
        "cp": [],
        "dtol_fraction": [],
        "correction_fraction": [],
        "ignore_check_fraction": [],
    }


################################################################


class Rhot_gaz_parfait_qc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas used with a quasi-compressible fluid where the state equation is
    defined as rho = f(T).
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure of the gas Cp.""",
        default=0.0,
    )
    prandtl: Optional[float] = Field(
        description=r"""Prandtl number of the gas Pr=mu*Cp/lambda""", default=None
    )
    rho_xyz: Optional[Field_base] = Field(
        description=r"""Defined with a Champ_Fonc_xyz to define a constant rho with time (space dependent)""",
        default=None,
    )
    rho_t: Optional[str] = Field(
        description=r"""Expression of T used to calculate rho. This can lead to a variable rho, both in space and in time.""",
        default=None,
    )
    t_min: Optional[float] = Field(
        description=r"""Temperature may, in some cases, locally and temporarily be very small (and negative) even though computation converges. T_min keyword allows to set a lower limit of temperature (in Kelvin, -1000 by default). WARNING: DO NOT USE THIS KEYWORD WITHOUT CHECKING CAREFULY YOUR RESULTS!""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "cp": [],
        "prandtl": [],
        "rho_xyz": [],
        "rho_t": [],
        "t_min": [],
    }


################################################################


class Perfect_gaz_qc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas state law used with a quasi-compressible fluid.
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    cv: Optional[float] = Field(
        description=r"""Specific heat at constant volume (J/kg/K).""", default=None
    )
    gamma: Optional[float] = Field(description=r"""Cp/Cv""", default=None)
    prandtl: float = Field(
        description=r"""Prandtl number of the gas Pr=mu*Cp/lambda""", default=0.0
    )
    rho_constant_pour_debug: Optional[Field_base] = Field(
        description=r"""For developers to debug the code with a constant rho.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["gaz_parfait_qc"],
        "cp": [],
        "cv": [],
        "gamma": [],
        "prandtl": [],
        "rho_constant_pour_debug": [],
    }


################################################################


class Binaire_gaz_parfait_qc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas binary mixtures state law used with a quasi-compressible fluid under
    the iso-thermal and iso-bar assumptions.
    """

    molar_mass1: float = Field(
        description=r"""Molar mass of species 1 (in kg/mol).""", default=0.0
    )
    molar_mass2: float = Field(
        description=r"""Molar mass of species 2 (in kg/mol).""", default=0.0
    )
    mu1: float = Field(
        description=r"""Dynamic viscosity of species 1 (in kg/m.s).""", default=0.0
    )
    mu2: float = Field(
        description=r"""Dynamic viscosity of species 2 (in kg/m.s).""", default=0.0
    )
    temperature: float = Field(
        description=r"""Temperature (in Kelvin) which will be constant during the simulation since this state law only works for iso-thermal conditions.""",
        default=0.0,
    )
    diffusion_coeff: float = Field(
        description=r"""Diffusion coefficient assumed the same for both species (in m2/s).""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "molar_mass1": [],
        "molar_mass2": [],
        "mu1": [],
        "mu2": [],
        "temperature": [],
        "diffusion_coeff": [],
    }


################################################################


class Eos_qc(Loi_etat_tppi_base):
    r"""
    Class for using EOS with QC problem
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    fluid: str = Field(description=r"""Fluid name in the EOS model""", default="")
    model: str = Field(description=r"""EOS model name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "cp": [], "fluid": [], "model": []}


################################################################


class Convection_diffusion_espece_multi_qc(Eqn_base):
    r"""
    Species conservation equation for a multi-species quasi-compressible fluid.
    """

    espece: Optional[Espece] = Field(
        description=r"""Assosciate a species (with its properties) to the equation""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "espece": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Navier_stokes_qc(Navier_stokes_standard):
    r"""
    Navier-Stokes equation for a quasi-compressible fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_thermohydraulique_qc(Pb_base):
    r"""
    Resolution of thermo-hydraulic problem for a quasi-compressible fluid.

    Keywords for the unknowns other than pressure, velocity, temperature are :

    masse_volumique : density

    enthalpie : enthalpy

    pression : reduced pressure

    pression_tot : total pressure.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    navier_stokes_qc: Navier_stokes_qc = Field(
        description=r"""Navier-Stokes equation for a quasi-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_qc()"),
    )
    convection_diffusion_chaleur_qc: Convection_diffusion_chaleur_qc = Field(
        description=r"""Temperature equation for a quasi-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_qc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_qc": [],
        "convection_diffusion_chaleur_qc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_hydraulique_melange_binaire_qc(Pb_base):
    r"""
    Resolution of a binary mixture problem for a quasi-compressible fluid with an iso-thermal
    condition.

    Keywords for the unknowns other than pressure, velocity, fraction_massique are :

    masse_volumique : density

    pression : reduced pressure

    pression_tot : total pressure.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""The various constituants associated to the problem.""",
        default=None,
    )
    navier_stokes_qc: Navier_stokes_qc = Field(
        description=r"""Navier-Stokes equation for a quasi-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_qc()"),
    )
    convection_diffusion_espece_binaire_qc: Convection_diffusion_espece_binaire_qc = Field(
        description=r"""Species conservation equation for a binary quasi-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_espece_binaire_qc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "constituant": [],
        "navier_stokes_qc": [],
        "convection_diffusion_espece_binaire_qc": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Mass_source(Interprete):
    r"""
    Mass source used in a dilatable simulation to add/reduce a mass at the boundary
    (volumetric source in the first cell of a given boundary).
    """

    bord: str = Field(
        description=r"""Name of the boundary where the source term is applied""",
        default="",
    )
    surfacic_flux: Front_field_base = Field(
        description=r"""The boundary field that the user likes to apply: for example, champ_front_uniforme, ch_front_input_uniform or champ_front_fonc_t""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "bord": [], "surfacic_flux": []}


################################################################


class Sortie_libre_temperature_imposee_h(Neumann):
    r"""
    Open boundary for heat equation with enthalpy as unknown.
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Frontiere_ouverte_rho_u_impose(Frontiere_ouverte_vitesse_imposee_sortie):
    r"""
    This keyword is used to designate a condition of imposed mass rate at an open boundary
    called bord (edge). The imposed mass rate field at the inlet is vectorial and the imposed
    velocity values are expressed in kg.s-1. This boundary condition can be used only with the
    Quasi compressible model.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Paroi_echange_externe_impose_h(Paroi_echange_externe_impose):
    r"""
    Particular case of class paroi_echange_externe_impose for enthalpy equation.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "h_or_t": ["h_imp"],
        "himpc": [],
        "t_or_h": ["text"],
        "ch": [],
    }


################################################################


class Entree_temperature_imposee_h(Frontiere_ouverte_temperature_imposee):
    r"""
    Particular case of class frontiere_ouverte_temperature_imposee for enthalpy equation.
    """

    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Pb_thermohydraulique_especes_qc(Pb_avec_passif):
    r"""
    Resolution of thermo-hydraulic problem for a multi-species quasi-compressible fluid.
    """

    fluide_quasi_compressible: Fluide_quasi_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_quasi_compressible()"),
    )
    navier_stokes_qc: Navier_stokes_qc = Field(
        description=r"""Navier-Stokes equation for a quasi-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_qc()"),
    )
    convection_diffusion_chaleur_qc: Convection_diffusion_chaleur_qc = Field(
        description=r"""Temperature equation for a quasi-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_qc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_qc": [],
        "convection_diffusion_chaleur_qc": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Fluide_weakly_compressible(Fluide_dilatable_base):
    r"""
    Weakly-compressible flow with a low mach number assumption; this means that the thermo-
    dynamic pressure (used in state law) can vary in space.
    """

    loi_etat: Optional[Loi_etat_base] = Field(
        description=r"""The state law that will be associated to the Weakly-compressible fluid.""",
        default=None,
    )
    sutherland: Optional[Bloc_sutherland] = Field(
        description=r"""Sutherland law for viscosity and for conductivity.""",
        default=None,
    )
    traitement_pth: Optional[Literal["constant"]] = Field(
        description=r"""Particular treatment for the thermodynamic pressure Pth ; there is currently one possibility:  1) the keyword \'constant\' makes it possible to have a constant Pth but not uniform in space ; it\'s the good choice when the flow is open (e.g. with pressure boundary conditions).""",
        default=None,
    )
    lambda_: Optional[Field_base] = Field(
        description=r"""Conductivity (W.m-1.K-1).""", default=None
    )
    mu: Optional[Field_base] = Field(
        description=r"""Dynamic viscosity (kg.m-1.s-1).""", default=None
    )
    pression_thermo: Optional[float] = Field(
        description=r"""Initial thermo-dynamic pressure used in the assosciated state law.""",
        default=None,
    )
    pression_xyz: Optional[Field_base] = Field(
        description=r"""Initial thermo-dynamic pressure used in the assosciated state law. It should be defined with as a Champ_Fonc_xyz.""",
        default=None,
    )
    use_total_pressure: Optional[int] = Field(
        description=r"""Flag (0 or 1) used to activate and use the total pressure in the assosciated state law. The default value of this Flag is 0.""",
        default=None,
    )
    use_hydrostatic_pressure: Optional[int] = Field(
        description=r"""Flag (0 or 1) used to activate and use the hydro-static pressure in the assosciated state law. The default value of this Flag is 0.""",
        default=None,
    )
    use_grad_pression_eos: Optional[int] = Field(
        description=r"""Flag (0 or 1) used to specify whether or not the gradient of the thermo-dynamic pressure will be taken into account in the source term of the temperature equation (case of a non-uniform pressure). The default value of this Flag is 1 which means that the gradient is used in the source.""",
        default=None,
    )
    time_activate_ptot: Optional[float] = Field(
        description=r"""Time (in seconds) at which the total pressure will be used in the assosciated state law.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "loi_etat": [],
        "sutherland": [],
        "traitement_pth": [],
        "lambda_": ["lambda_u", "lambda"],
        "mu": [],
        "pression_thermo": [],
        "pression_xyz": [],
        "use_total_pressure": [],
        "use_hydrostatic_pressure": [],
        "use_grad_pression_eos": [],
        "time_activate_ptot": [],
        "indice": [],
        "kappa": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "cp": [],
    }


################################################################


class Navier_stokes_wc(Navier_stokes_standard):
    r"""
    Navier-Stokes equation for a weakly-compressible fluid.
    """

    mass_source: Optional[Mass_source] = Field(
        description=r"""Mass source used in a dilatable simulation to add/reduce a mass at the boundary (volumetric source in the first cell of a given boundary).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "mass_source": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_chaleur_wc(Eqn_base):
    r"""
    Temperature equation for a weakly-compressible fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_thermohydraulique_especes_wc(Pb_avec_passif):
    r"""
    Resolution of thermo-hydraulic problem for a multi-species weakly-compressible fluid.
    """

    fluide_weakly_compressible: Fluide_weakly_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_weakly_compressible()"),
    )
    navier_stokes_wc: Navier_stokes_wc = Field(
        description=r"""Navier-Stokes equation for a weakly-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_wc()"),
    )
    convection_diffusion_chaleur_wc: Convection_diffusion_chaleur_wc = Field(
        description=r"""Temperature equation for a weakly-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_wc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_weakly_compressible": [],
        "navier_stokes_wc": [],
        "convection_diffusion_chaleur_wc": [],
        "equations_scalaires_passifs": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Binaire_gaz_parfait_wc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas binary mixtures state law used with a weakly-compressible fluid
    under the iso-thermal and iso-bar assumptions.
    """

    molar_mass1: float = Field(
        description=r"""Molar mass of species 1 (in kg/mol).""", default=0.0
    )
    molar_mass2: float = Field(
        description=r"""Molar mass of species 2 (in kg/mol).""", default=0.0
    )
    mu1: float = Field(
        description=r"""Dynamic viscosity of species 1 (in kg/m.s).""", default=0.0
    )
    mu2: float = Field(
        description=r"""Dynamic viscosity of species 2 (in kg/m.s).""", default=0.0
    )
    temperature: float = Field(
        description=r"""Temperature (in Kelvin) which will be constant during the simulation since this state law only works for iso-thermal conditions.""",
        default=0.0,
    )
    diffusion_coeff: float = Field(
        description=r"""Diffusion coefficient assumed the same for both species (in m2/s).""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "molar_mass1": [],
        "molar_mass2": [],
        "mu1": [],
        "mu2": [],
        "temperature": [],
        "diffusion_coeff": [],
    }


################################################################


class Eos_wc(Loi_etat_tppi_base):
    r"""
    Class for using EOS with WC problem
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    fluid: str = Field(description=r"""Fluid name in the EOS model""", default="")
    model: str = Field(description=r"""EOS model name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "cp": [], "fluid": [], "model": []}


################################################################


class Perfect_gaz_wc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas state law used with a weakly-compressible fluid.
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    cv: Optional[float] = Field(
        description=r"""Specific heat at constant volume (J/kg/K).""", default=None
    )
    gamma: Optional[float] = Field(description=r"""Cp/Cv""", default=None)
    prandtl: float = Field(
        description=r"""Prandtl number of the gas Pr=mu*Cp/lambda""", default=0.0
    )
    _synonyms: ClassVar[dict] = {
        None: ["gaz_parfait_wc"],
        "cp": [],
        "cv": [],
        "gamma": [],
        "prandtl": [],
    }


################################################################


class Coolprop_wc(Loi_etat_tppi_base):
    r"""
    Class for using CoolProp with WC problem
    """

    cp: float = Field(
        description=r"""Specific heat at constant pressure (J/kg/K).""", default=0.0
    )
    fluid: str = Field(description=r"""Fluid name in the CoolProp model""", default="")
    model: str = Field(description=r"""CoolProp model name""", default="")
    _synonyms: ClassVar[dict] = {None: [], "cp": [], "fluid": [], "model": []}


################################################################


class Multi_gaz_parfait_wc(Loi_etat_gaz_parfait_base):
    r"""
    Class for perfect gas multi-species mixtures state law used with a weakly-compressible
    fluid.
    """

    species_number: int = Field(
        description=r"""Number of species you are considering in your problem.""",
        default=0,
    )
    diffusion_coeff: Field_base = Field(
        description=r"""Diffusion coefficient of each species, defined with a Champ_uniforme of dimension equals to the species_number.""",
        default_factory=lambda: eval("Field_base()"),
    )
    molar_mass: Field_base = Field(
        description=r"""Molar mass of each species, defined with a Champ_uniforme of dimension equals to the species_number.""",
        default_factory=lambda: eval("Field_base()"),
    )
    mu: Field_base = Field(
        description=r"""Dynamic viscosity of each species, defined with a Champ_uniforme of dimension equals to the species_number.""",
        default_factory=lambda: eval("Field_base()"),
    )
    cp: Field_base = Field(
        description=r"""Specific heat at constant pressure of the gas Cp, defined with a Champ_uniforme of dimension equals to the species_number..""",
        default_factory=lambda: eval("Field_base()"),
    )
    prandtl: float = Field(
        description=r"""Prandtl number of the gas Pr=mu*Cp/lambda.""", default=0.0
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "species_number": [],
        "diffusion_coeff": [],
        "molar_mass": [],
        "mu": [],
        "cp": [],
        "prandtl": [],
    }


################################################################


class Convection_diffusion_espece_binaire_wc(Eqn_base):
    r"""
    Species conservation equation for a binary weakly-compressible fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Convection_diffusion_espece_multi_wc(Eqn_base):
    r"""
    Species conservation equation for a multi-species weakly-compressible fluid.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_hydraulique_melange_binaire_wc(Pb_base):
    r"""
    Resolution of a binary mixture problem for a weakly-compressible fluid with an iso-thermal
    condition.

    Keywords for the unknowns other than pressure, velocity, fraction_massique are :

    masse_volumique : density

    pression : reduced pressure

    pression_tot : total pressure

    pression_hydro : hydro-static pressure

    pression_eos : pressure used in state equation.
    """

    fluide_weakly_compressible: Fluide_weakly_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_weakly_compressible()"),
    )
    navier_stokes_wc: Navier_stokes_wc = Field(
        description=r"""Navier-Stokes equation for a weakly-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_wc()"),
    )
    convection_diffusion_espece_binaire_wc: Convection_diffusion_espece_binaire_wc = Field(
        description=r"""Species conservation equation for a binary weakly-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_espece_binaire_wc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_weakly_compressible": [],
        "navier_stokes_wc": [],
        "convection_diffusion_espece_binaire_wc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_thermohydraulique_wc(Pb_base):
    r"""
    Resolution of thermo-hydraulic problem for a weakly-compressible fluid.

    Keywords for the unknowns other than pressure, velocity, temperature are :

    masse_volumique : density

    pression : reduced pressure

    pression_tot : total pressure

    pression_hydro : hydro-static pressure

    pression_eos : pressure used in state equation.
    """

    fluide_weakly_compressible: Fluide_weakly_compressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_weakly_compressible()"),
    )
    navier_stokes_wc: Navier_stokes_wc = Field(
        description=r"""Navier-Stokes equation for a weakly-compressible fluid.""",
        default_factory=lambda: eval("Navier_stokes_wc()"),
    )
    convection_diffusion_chaleur_wc: Convection_diffusion_chaleur_wc = Field(
        description=r"""Temperature equation for a weakly-compressible fluid.""",
        default_factory=lambda: eval("Convection_diffusion_chaleur_wc()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_weakly_compressible": [],
        "navier_stokes_wc": [],
        "convection_diffusion_chaleur_wc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Launder_sharmma(Modele_fonction_bas_reynolds_base):
    r"""
    Model described in ' Launder, B. E. and Sharma, B. I. (1974), Application of the Energy-
    Dissipation Model of Turbulence to the Calculation of Flow Near a Spinning Disc, Letters
    in Heat and Mass Transfer, Vol. 1, No. 2, pp. 131-138.'
    """

    _synonyms: ClassVar[dict] = {None: ["launder_sharma"]}


################################################################


class Cond_lim_k_simple_flux_nul(Condlim_base):
    r"""
    Adaptive wall law boundary condition for turbulent kinetic energy
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Transport_k_eps_realisable(Eqn_base):
    r"""
    Realizable K-Epsilon Turbulence Model Transport Equations for K and Epsilon.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Bloc_rho_fonc_c(Objet_lecture):
    r"""
    if rho has a general form
    """

    champ_fonc_fonction: Optional[Literal["champ_fonc_fonction"]] = Field(
        description=r"""Champ_Fonc_Fonction""", default=None
    )
    problem_name: Optional[str] = Field(
        description=r"""Name of problem.""", default=None
    )
    concentration: Optional[Literal["concentration"]] = Field(
        description=r"""concentration""", default=None
    )
    dim: Optional[int] = Field(
        description=r"""dimension of the problem""", default=None
    )
    val: Optional[str] = Field(description=r"""function of rho""", default=None)
    champ_uniforme: Optional[Literal["champ_uniforme"]] = Field(
        description=r"""Champ_Uniforme""", default=None
    )
    fielddim: Optional[int] = Field(
        description=r"""dimension of the problem""", default=None
    )
    val2: Optional[str] = Field(description=r"""function of rho""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "champ_fonc_fonction": [],
        "problem_name": [],
        "concentration": [],
        "dim": [],
        "val": [],
        "champ_uniforme": [],
        "fielddim": [],
        "val2": [],
    }


################################################################


class Bloc_boussinesq(Objet_lecture):
    r"""
    choice of rho formulation
    """

    probleme: Optional[str] = Field(description=r"""Name of problem.""", default=None)
    rho_1: Optional[float] = Field(description=r"""value of rho""", default=None)
    rho_2: Optional[float] = Field(description=r"""value of rho""", default=None)
    rho_fonc_c: Optional[Bloc_rho_fonc_c] = Field(
        description=r"""to use for define a general form for rho""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "probleme": [],
        "rho_1": [],
        "rho_2": [],
        "rho_fonc_c": ["rho_fonc_c_"],
    }


################################################################


class Approx_boussinesq(Objet_lecture):
    r"""
    different mass density formulation are available depending if the Boussinesq approximation
    is made or not
    """

    yes_or_no: Literal["oui", "non"] = Field(
        description=r"""To use or not the Boussinesq approximation.""", default="oui"
    )
    bloc_bouss: Bloc_boussinesq = Field(
        description=r"""to choose the rho formulation""",
        default_factory=lambda: eval("Bloc_boussinesq()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "yes_or_no": [], "bloc_bouss": []}


################################################################


class Bloc_mu_fonc_c(Objet_lecture):
    r"""
    if mu has a general form
    """

    champ_fonc_fonction: Optional[Literal["champ_fonc_fonction"]] = Field(
        description=r"""Champ_Fonc_Fonction""", default=None
    )
    problem_name: Optional[str] = Field(
        description=r"""Name of problem.""", default=None
    )
    concentration: Optional[Literal["concentration"]] = Field(
        description=r"""concentration""", default=None
    )
    dim: Optional[int] = Field(
        description=r"""dimension of the problem""", default=None
    )
    val: Optional[str] = Field(description=r"""function of mu""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "champ_fonc_fonction": [],
        "problem_name": [],
        "concentration": [],
        "dim": [],
        "val": [],
    }


################################################################


class Bloc_visco2(Objet_lecture):
    r"""
    choice of mu formulation
    """

    probleme: Optional[str] = Field(description=r"""Name of problem.""", default=None)
    mu_1: Optional[float] = Field(description=r"""value of mu""", default=None)
    mu_2: Optional[float] = Field(description=r"""value of mu""", default=None)
    mu_fonc_c: Optional[Bloc_mu_fonc_c] = Field(
        description=r"""to use for define a general form for mu""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "probleme": [],
        "mu_1": [],
        "mu_2": [],
        "mu_fonc_c": ["mu_fonc_c_"],
    }


################################################################


class Visco_dyn_cons(Objet_lecture):
    r"""
    different treatment of the kinematic viscosity could be done depending of the use of the
    Boussinesq approximation or the constant dynamic viscosity approximation
    """

    yes_or_no: Literal["oui", "non"] = Field(
        description=r"""To use or not the constant dynamic viscosity""", default="oui"
    )
    bloc_visco: Bloc_visco2 = Field(
        description=r"""to choose the mu formulation""",
        default_factory=lambda: eval("Bloc_visco2()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "yes_or_no": [], "bloc_visco": []}


################################################################


class Navier_stokes_phase_field(Navier_stokes_standard):
    r"""
    Navier Stokes equation for the Phase Field problem.
    """

    approximation_de_boussinesq: Optional[Approx_boussinesq] = Field(
        description=r"""To use or not the Boussinesq approximation.""", default=None
    )
    viscosite_dynamique_constante: Optional[Visco_dyn_cons] = Field(
        description=r"""To use or not a viscosity which will depends on concentration C (in fact, C is the unknown of Cahn-Hilliard equation).""",
        default=None,
    )
    gravite: Optional[List[float]] = Field(
        description=r"""Keyword to define gravity in the case Boussinesq approximation is not used.""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "approximation_de_boussinesq": [],
        "viscosite_dynamique_constante": [],
        "gravite": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Modele_fonc_realisable_base(Class_generic):
    r"""
    Base class for Functions necessary to Realizable K-Epsilon Turbulence Model
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class K_epsilon_realisable_bicephale(Mod_turb_hyd_rans):
    r"""
    Realizable Two-headed K-Epsilon Turbulence Model
    """

    transport_k: Optional[str] = Field(
        description=r"""Keyword to define the realisable (k) transportation equation.""",
        default=None,
    )
    transport_epsilon: Optional[str] = Field(
        description=r"""Keyword to define the realisable (eps) transportation equation.""",
        default=None,
    )
    modele_fonc_realisable: Optional[Modele_fonc_realisable_base] = Field(
        description=r"""This keyword is used to set the model used""", default=None
    )
    prandtl_k: float = Field(
        description=r"""Keyword to change the Prk value (default 1.0).""", default=0.0
    )
    prandtl_eps: float = Field(
        description=r"""Keyword to change the Pre value (default 1.3)""", default=0.0
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "transport_k": [],
        "transport_epsilon": [],
        "modele_fonc_realisable": [],
        "prandtl_k": [],
        "prandtl_eps": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Terme_dissipation_energie_cinetique_turbulente(Source_base):
    r"""
    Dissipation source term used in the TKE equation
    """

    beta_k: Optional[float] = Field(
        description=r"""Constant for the used model""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "beta_k": []}


################################################################


class Production_echelle_temp_taux_diss_turb(Source_base):
    r"""
    Production source term used in the tau and omega equations
    """

    alpha_omega: Optional[float] = Field(
        description=r"""Constant for the used model""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "alpha_omega": []}


################################################################


class Dissipation_echelle_temp_taux_diss_turb(Source_base):
    r"""
    Dissipation source term used in the tau and omega equations
    """

    beta_omega: Optional[float] = Field(
        description=r"""Constant for the used model""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "beta_omega": []}


################################################################


class Diffusion_croisee_echelle_temp_taux_diss_turb(Source_base):
    r"""
    Cross-diffusion source term used in the tau and omega equations
    """

    sigma_d: Optional[float] = Field(
        description=r"""Constant for the used model""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "sigma_d": []}


################################################################


class Listdeuxmots_sacc(Listobj):
    r"""
    List of groups of two words (without curly brackets).
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Source_robin_scalaire(Source_base):
    r"""
    This source term should be used when a Paroi_decalee_Robin boundary condition is set in a
    an energy equation. The source term will be applied on the N specified boundaries. The
    values temp_wall_valueI are the temperature specified on the Ith boundary. The last value
    dt_impr is a printing period which is mandatory to specify in the data file but has no
    effect yet.
    """

    bords: Annotated[List[Deuxmots], "Listdeuxmots_sacc"] = Field(
        description=r"""List of groups of two words (without curly brackets).""",
        default_factory=list,
    )
    _synonyms: ClassVar[dict] = {None: [], "bords": []}


################################################################


class Triple_line_model_ft_disc(Objet_u):
    r"""
    Triple Line Model (TCL)
    """

    qtcl: Optional[float] = Field(
        description=r"""Heat flux contribution to micro-region [W/m]""", default=None
    )
    lv: Optional[float] = Field(description=r"""Slip length (unused)""", default=None)
    coeffa: Optional[float] = Field(description=r"""not_set""", default=None)
    coeffb: Optional[float] = Field(description=r"""not_set""", default=None)
    theta_app: Optional[float] = Field(
        description=r"""Apparent contact angle (Cox-Voinov)""", default=None
    )
    ylim: Optional[float] = Field(description=r"""not_set""", default=None)
    ym: Optional[float] = Field(
        description=r"""Wall distance of the point M delimiting micro/meso transition [m]""",
        default=None,
    )
    sm: float = Field(
        description=r"""Curvilinear abscissa of the point M delimiting micro/meso transition [m]""",
        default=0.0,
    )
    hydraulic_equation: str = Field(
        description=r"""Hydraulic equation name""", default=""
    )
    thermal_equation: str = Field(description=r"""Thermal equation name""", default="")
    interface_equation: str = Field(
        description=r"""Interface equation name""", default=""
    )
    ymeso: Optional[float] = Field(
        description=r"""Meso region extension in wall-normal direction [m]""",
        default=None,
    )
    n_extend_meso: Optional[int] = Field(
        description=r"""Meso region extension in number of cells [-]""", default=None
    )
    initial_cl_xcoord: Optional[float] = Field(
        description=r"""Initial interface position (unused)""", default=None
    )
    rc_tcl_gridn: Optional[float] = Field(
        description=r"""Radius of nucleate site; [in number of grids]""", default=None
    )
    thetac_tcl: Optional[float] = Field(
        description=r"""imposed contact angle [in degree] to force bubble pinching / necking once TCL entre nucleate site""",
        default=None,
    )
    reinjection_tcl: Optional[bool] = Field(
        description=r"""This flag activates the automatic injection of a new nucleate seed with a specified shape when the temperature in the nucleation site becomes higher than a certain threshold (tempC_tcl). The shape of the seed is determined by the radius Rc_tcl_GridN and the contact angle thetaC_tcl. The nucleation site is considered free when there are no bubbles present. The site size is defined by Rc_tcl_GridN. This temperature threshold, termed tempC_tcl, is the activation temperature. Setting this temperature implies a wall temperature, therefore, activating reinjection_tcl is ONLY possible for a simulation coupled with solid conduction.  When reinjection_tcl is activated, the values of tempC_tcl (default 10K), Rc_tcl_GridN (default 4 grid sizes), and thetaC_tcl (default 150 degrees) should be provided. Unless (STRONGLY not recommended), the default values (indicated in parentheses) will be used.  If reinjection_tcl is not activated (by default), the mechanism of Numerically forcing bubble pinching/necking will be used for multi-cycle simulation. Once the Triple Contact Line (TCL) enters the nucleation site, a big contact angle thetaC_tcl is imposed to initiate bubble pinching/necking. After the bubble pinching ends, the large bubble above will depart, leaving the remaining part to serve as the nucleate seed. This process is equivalent to immediately inserting a new seed with a prescribed shape (determined by the nucleation site size and contact angle) once a bubble departs. Site size is defined by Rc_tcl_GridN (default 4 grid sizes). Contact angle thetaC_tcl (default 150 degrees). Useful for a standalone (not coupling with solid conduction) simulation.""",
        default=None,
    )
    distri_first_facette: Optional[bool] = Field(
        description=r"""This flag determines whether to distribute the Qtcl into all grids occupied by the first facette according to their area proportions. When set, the flux is redistributed into all grids occupied by the first facette based on their area proportions. Default value is 0, the flux is distributed differently: similar to the Meso zone, it is only distributed to grids within the Micro-zone (where the height of the front y is smaller than the size of Micro ym). The distribution of this flux is logarithmically proportional to y between 5.6nm (here interpreted as the value 0 in logarithm) and ym. In practice, in most cases, it will distribute all the flux locally in the first grid.""",
        default=None,
    )
    file_name: Optional[float] = Field(
        description=r"""Input file to set TCL model""", default=None
    )
    deactivate: Optional[bool] = Field(
        description=r"""Simple way to disable completely the TCL model contribution""",
        default=None,
    )
    inout_method: Optional[Literal["exact", "approx", "both"]] = Field(
        description=r"""Type of method for in out calc. By defautl, exact method is used""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "qtcl": [],
        "lv": [],
        "coeffa": [],
        "coeffb": [],
        "theta_app": [],
        "ylim": [],
        "ym": [],
        "sm": [],
        "hydraulic_equation": ["equation_navier_stokes"],
        "thermal_equation": ["equation_temperature"],
        "interface_equation": ["equation_interface"],
        "ymeso": [],
        "n_extend_meso": [],
        "initial_cl_xcoord": [],
        "rc_tcl_gridn": [],
        "thetac_tcl": [],
        "reinjection_tcl": [],
        "distri_first_facette": [],
        "file_name": [],
        "deactivate": [],
        "inout_method": [],
    }


################################################################


class Source_robin(Source_base):
    r"""
    This source term should be used when a Paroi_decalee_Robin boundary condition is set in a
    hydraulic equation. The source term will be applied on the N specified boundaries. To
    post-process the values of tauw, u_tau and Reynolds_tau into the files tauw_robin.dat,
    reynolds_tau_robin.dat and u_tau_robin.dat, you must add a block Traitement_particulier {
    canal { } }
    """

    bords: Annotated[List[Nom_anonyme], "Vect_nom"] = Field(
        description=r"""Vect of name.""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "bords": []}


################################################################


class Transport_k_omega(Eqn_base):
    r"""
    The (k-omega) transport equation.
    """

    with_nu: Optional[Literal["yes", "no"]] = Field(
        description=r"""yes/no (default no)""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "with_nu": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class K_omega(Mod_turb_hyd_rans):
    r"""
    Turbulence model (k-omega).
    """

    transport_k_omega: Optional[Transport_k_omega] = Field(
        description=r"""Keyword to define the (k-omega) transportation equation.""",
        default=None,
    )
    model_variant: Optional[str] = Field(
        description=r"""Model variant for k-omega (default value STD)""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "transport_k_omega": [],
        "model_variant": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Injection_qdm_nulle(Source_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Paroi_frottante_loi(Condlim_base):
    r"""
    Adaptive wall-law boundary condition for velocity
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Type_diffusion_turbulente_multiphase_k_omega(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    not_set
    """

    limiter: Optional[str] = Field(description=r"""not_set""", default=None)
    sigma: Optional[float] = Field(description=r"""not_set""", default=None)
    beta_k: Optional[float] = Field(description=r"""not_set""", default=None)
    gas_turb: Optional[bool] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["k_omega"],
        "limiter": ["limiteur"],
        "sigma": [],
        "beta_k": [],
        "gas_turb": [],
    }


################################################################


class Ceg_areva(Objet_lecture):
    r"""
    not_set
    """

    c: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: ["nul"], "c": []}


################################################################


class Ceg_cea_jaea(Objet_lecture):
    r"""
    not_set
    """

    normalise: Optional[int] = Field(
        description=r"""renormalize (1) or not (0) values alpha and gamma""",
        default=None,
    )
    nb_mailles_mini: Optional[int] = Field(
        description=r"""Sets the minimum number of cells for the detection of a vortex.""",
        default=None,
    )
    min_critere_q_sur_max_critere_q: Optional[float] = Field(
        description=r"""Is an optional keyword used to correct the minimum values of Q's criterion taken into account in the detection of a vortex""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "normalise": [],
        "nb_mailles_mini": [],
        "min_critere_q_sur_max_critere_q": [],
    }


################################################################


class Traitement_particulier_ceg(Traitement_particulier_base):
    r"""
    Keyword for a CEG ( Gas Entrainment Criteria) calculation. An objective is deepening gas
    entrainment on the free surface. Numerical analysis can be performed to predict the
    hydraulic and geometric conditions that can handle gas entrainment from the free surface.
    """

    frontiere: str = Field(
        description=r"""To specify the boundaries conditions representing the free surfaces""",
        default="",
    )
    t_deb: float = Field(
        description=r"""value of the CEG's initial calculation time""", default=0.0
    )
    t_fin: Optional[float] = Field(
        description=r"""not_set time during which the CEG's calculation was stopped""",
        default=None,
    )
    dt_post: Optional[float] = Field(
        description=r"""periode refers to the printing period, this value is expressed in seconds""",
        default=None,
    )
    haspi: float = Field(
        description=r"""The suction height required to calculate AREVA's criterion""",
        default=0.0,
    )
    debug: Optional[int] = Field(description=r"""not_set""", default=None)
    areva: Optional[Ceg_areva] = Field(
        description=r"""AREVA's criterion""", default=None
    )
    cea_jaea: Optional[Ceg_cea_jaea] = Field(
        description=r"""CEA_JAEA's criterion""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["ceg"],
        "frontiere": [],
        "t_deb": [],
        "t_fin": [],
        "dt_post": [],
        "haspi": [],
        "debug": [],
        "areva": [],
        "cea_jaea": [],
    }


################################################################


class Pb_rayo_conduction(Pb_conduction):
    r"""
    Resolution of the heat equation with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "solide": [],
        "conduction": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_hydraulique(Pb_hydraulique):
    r"""
    Resolution of the Navier-Stokes equations with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_standard": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_thermohydraulique(Pb_thermohydraulique):
    r"""
    Resolution of pb_thermohydraulique with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "fluide_ostwald": [],
        "fluide_sodium_liquide": [],
        "fluide_sodium_gaz": [],
        "correlations": [],
        "navier_stokes_standard": [],
        "convection_diffusion_temperature": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_thermohydraulique_qc(Pb_thermohydraulique_qc):
    r"""
    Resolution of pb_thermohydraulique_QC with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_qc": [],
        "convection_diffusion_chaleur_qc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_hydraulique_turbulent(Pb_hydraulique_turbulent):
    r"""
    Resolution of pb_hydraulique_turbulent with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_thermohydraulique_turbulent(Pb_thermohydraulique_turbulent):
    r"""
    Resolution of pb_thermohydraulique_turbulent with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_turbulent": [],
        "convection_diffusion_temperature_turbulent": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Pb_rayo_thermohydraulique_turbulent_qc(Pb_thermohydraulique_turbulent_qc):
    r"""
    Resolution of pb_thermohydraulique_turbulent_qc with rayonnement.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_quasi_compressible": [],
        "navier_stokes_turbulent_qc": [],
        "convection_diffusion_chaleur_turbulent_qc": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Mod_turb_hyd_rans_keps(Mod_turb_hyd_rans):
    r"""
    Class for RANS turbulence model for Navier-Stokes equations.
    """

    eps_min: Optional[float] = Field(
        description=r"""Lower limitation of epsilon (default value 1.e-10).""",
        default=None,
    )
    eps_max: Optional[float] = Field(
        description=r"""Upper limitation of epsilon (default value 1.e+10).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "eps_min": [],
        "eps_max": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Source_bif(Source_base):
    r"""
    Additional fluctuations induced by the movement of bubbles, only available in PolyMAC_P0
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Cond_lim_omega_dix(Condlim_base):
    r"""
    Adaptive wall law boundary condition for turbulent dissipation rate
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Cond_lim_omega_demi(Condlim_base):
    r"""
    Adaptive wall law boundary condition for turbulent dissipation rate
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Pb_hydraulique_ale(Pb_base):
    r"""
    Resolution of hydraulic problems for ALE
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_standard_ale: Navier_stokes_standard = Field(
        description=r"""Navier-Stokes equations for ALE problems""",
        default_factory=lambda: eval("Navier_stokes_standard()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_standard_ale": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Mod_turb_hyd_rans_bicephale(Mod_turb_hyd_rans):
    r"""
    Class for RANS turbulence model for Navier-Stokes equations.
    """

    eps_min: Optional[float] = Field(
        description=r"""Lower limitation of epsilon (default value 1.e-10).""",
        default=None,
    )
    eps_max: Optional[float] = Field(
        description=r"""Upper limitation of epsilon (default value 1.e+10).""",
        default=None,
    )
    prandtl_k: Optional[float] = Field(
        description=r"""Keyword to change the Prk value (default 1.0).""", default=None
    )
    prandtl_eps: Optional[float] = Field(
        description=r"""Keyword to change the Pre value (default 1.3)""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "eps_min": [],
        "eps_max": [],
        "prandtl_k": [],
        "prandtl_eps": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Champ_front_ale_beam(Front_field_base):
    r"""
    Class to define a Beam on a FSI boundary.
    """

    val: List[str] = Field(description=r""" Example: 3 0 0 0""", default_factory=list)
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Convection_sensibility(Convection_deriv):
    r"""
    A convective scheme for the sensibility problem.
    """

    opconv: Bloc_convection = Field(
        description=r"""Choice between: amont and muscl  Example: convection { Sensibility { amont } }""",
        default_factory=lambda: eval("Bloc_convection()"),
    )
    _synonyms: ClassVar[dict] = {None: ["sensibility"], "opconv": []}


################################################################


class Navier_stokes_std_ale(Navier_stokes_standard):
    r"""
    Resolution of hydraulic Navier-Stokes eq. on mobile domain (ALE)
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Production_hzdr(Source_base):
    r"""
    Additional source terms in the turbulent kinetic energy equation to model the fluctuations
    induced by bubbles.
    """

    constante_gravitation: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    c_k: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "constante_gravitation": [], "c_k": []}


################################################################


class Thermique(Listobj):
    r"""
    to add energy equation resolution if needed
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Thermique_bloc(Interprete):
    r"""
    not_set
    """

    cp_liquid: float = Field(
        description=r"""Liquid specific heat at constant pressure""", default=0.0
    )
    lambda_liquid: float = Field(
        description=r"""Liquid thermal conductivity""", default=0.0
    )
    cp_vapor: float = Field(
        description=r"""Vapor specific heat at constant pressure""", default=0.0
    )
    lambda_vapor: float = Field(
        description=r"""Vapor thermal conductivity""", default=0.0
    )
    fo: Optional[float] = Field(description=r"""not_set""", default=None)
    boundary_conditions: Bloc_lecture = Field(
        description=r"""boundary conditions""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    expression_t_init: Optional[str] = Field(
        description=r"""Expression of initial temperature (parser of x,y,z)""",
        default=None,
    )
    conv_temperature_negligible: Optional[bool] = Field(
        description=r"""neglect temperature convection""", default=None
    )
    type_temperature_convection_op: Optional[
        Literal["amont", "quick", "centre2", "centre4"]
    ] = Field(description=r"""convection operator""", default=None)
    diff_temp_negligible: Optional[bool] = Field(
        description=r"""neglect temperature diffusion""", default=None
    )
    wall_flux: Optional[bool] = Field(description=r"""not_set""", default=None)
    expression_t_ana: Optional[str] = Field(
        description=r"""Analytical expression T=f(x,y,z,t) for post-processing only""",
        default=None,
    )
    type_t_source: Optional[Literal["dabiri", "patch_dabiri", "unweighted_dabiri"]] = (
        Field(description=r"""source term""", default=None)
    )
    expression_source_temperature: Optional[str] = Field(
        description=r"""source terms""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "cp_liquid": [],
        "lambda_liquid": [],
        "cp_vapor": [],
        "lambda_vapor": [],
        "fo": [],
        "boundary_conditions": [],
        "expression_t_init": [],
        "conv_temperature_negligible": [],
        "type_temperature_convection_op": [],
        "diff_temp_negligible": [],
        "wall_flux": [],
        "expression_t_ana": [],
        "type_t_source": [],
        "expression_source_temperature": [],
    }


################################################################


class Projection_ale_boundary(Interprete):
    r"""
    block to compute the projection of a modal function on a mobile boundary. Use to compute
    modal added coefficients in FSI.
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    bloc: Bloc_lecture = Field(
        description=r"""between the braces, you must specify the numbers of the mobile borders then list these mobile borders and indicate the modal function which must be projected on these boundaries.  Example: Projection_ALE_boundary dom_name { 1 boundary_name 3 0.sin(pi*x)*1.e-4 0. }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Mod_turb_hyd_rans_komega(Mod_turb_hyd_rans):
    r"""
    Class for RANS turbulence model for Navier-Stokes equations.
    """

    omega_min: Optional[float] = Field(
        description=r"""Lower limitation of omega (default value 1.e-20).""",
        default=None,
    )
    omega_max: Optional[float] = Field(
        description=r"""Upper limitation of omega (default value 1.e+10).""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "omega_min": [],
        "omega_max": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Implicit_euler_steady_scheme(Schema_implicite_base):
    r"""
    This is the Implicit Euler scheme using a dual time step procedure (using local and global
    dt) for steady problems. Remark: the only possible solver choice for this scheme is the
    implicit_steady solver.
    """

    max_iter_implicite: Optional[int] = Field(
        description=r"""Maximum number of iterations allowed for the solver (by default 200)""",
        default=None,
    )
    steady_security_facteur: Optional[float] = Field(
        description=r"""Parameter used in the local time step calculation procedure in order to increase or decrease the local dt value (by default 0.5). We expect a strictly positive value""",
        default=None,
    )
    steady_global_dt: Optional[float] = Field(
        description=r"""This is the global time step used in the dual time step algorithm (by default 100). We expect a strictly positive value""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: ["schema_euler_implicite_stationnaire"],
        "max_iter_implicite": [],
        "steady_security_facteur": [],
        "steady_global_dt": [],
        "solveur": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Production_energie_cin_turb(Source_base):
    r"""
    Production source term for the TKE equation
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Navier_stokes_aposteriori(Navier_stokes_standard):
    r"""
    Modification of the Navier_Stokes_standard class in order to accept the
    estimateur_aposteriori post-processing. To post-process estimateur_aposteriori, add this
    keyword into the list of fields to be post-processed. This estimator whill generate a map
    of aposteriori error estimators; it is defined on each mesh cell and is a measure of the
    local discretisation error. This will serve for adaptive mesh refinement
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_hydraulique_aposteriori(Pb_base):
    r"""
    Modification of the pb_hydraulique problem in order to accept the estimateur_aposteriori
    post-processing.
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_aposteriori: Navier_stokes_aposteriori = Field(
        description=r"""Modification of the Navier_Stokes_standard class in order to accept the estimateur_aposteriori post-processing. To post-process estimateur_aposteriori, add this keyword into the list of fields to be post-processed. This estimator whill generate a map of aposteriori error estimators; it is defined on each mesh cell and is a measure of the local discretisation error. This will serve for adaptive mesh refinement""",
        default_factory=lambda: eval("Navier_stokes_aposteriori()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_aposteriori": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Navier_stokes_turbulent_ale(Navier_stokes_std_ale):
    r"""
    Resolution of hydraulic turbulent Navier-Stokes eq. on mobile domain (ALE)
    """

    modele_turbulence: Optional[Modele_turbulence_hyd_deriv] = Field(
        description=r"""Turbulence model for Navier-Stokes equations.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "modele_turbulence": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_hydraulique_turbulent_ale(Pb_base):
    r"""
    Resolution of hydraulic turbulent problems for ALE
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_turbulent_ale: Navier_stokes_turbulent_ale = Field(
        description=r"""Navier-Stokes_ALE equations as well as the associated turbulence model equations on mobile domain (ALE)""",
        default_factory=lambda: eval("Navier_stokes_turbulent_ale()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_turbulent_ale": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Implicite_ale(Implicite):
    r"""
    Implicite solver used for ALE problem
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Bloc_lecture_structural_dynamic_mesh_model(Objet_lecture):
    r"""
    bloc
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    mfront_library: Literal["mfront_library"] = Field(
        description=r"""Keyword to specify the path_to_libBehaviour.so""",
        default="mfront_library",
    )
    mfront_model_name: Literal["mfront_model_name"] = Field(
        description=r"""keyword to specify the Mfront model. Choice between Ogden and SaintVenantKirchhoffElasticity.""",
        default="mfront_model_name",
    )
    mfront_material_property: Literal["mfront_material_property"] = Field(
        description=r"""keyword to specify the material property. Eg. Ogden_alpha_, Ogden_mu_, Ogden_K""",
        default="mfront_material_property",
    )
    youngmodulus: Optional[float] = Field(description=r"""Young Module""", default=None)
    density: Optional[float] = Field(
        description=r"""fictitious structural density""", default=None
    )
    inertial_damping: Optional[float] = Field(
        description=r"""fictitious structural inertial damping""", default=None
    )
    grid_dt_min: Optional[float] = Field(
        description=r"""fictitious structural time step""", default=None
    )
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "aco": [],
        "mfront_library": [],
        "mfront_model_name": ["mfront_model"],
        "mfront_material_property": [],
        "youngmodulus": ["young"],
        "density": ["rho"],
        "inertial_damping": [],
        "grid_dt_min": [],
        "acof": [],
    }


################################################################


class Structural_dynamic_mesh_model(Interprete):
    r"""
    Fictitious structural model for mesh motion. Link with MGIS library
    """

    dom: str = Field(description=r"""domain name""", default="")
    bloc: Bloc_lecture_structural_dynamic_mesh_model = Field(
        description=r"""not_set""",
        default_factory=lambda: eval("Bloc_lecture_structural_dynamic_mesh_model()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Scheme_euler_explicite_ale(Schema_temps_base):
    r"""
    This is the Euler explicit scheme used for ALE problems.
    """

    _synonyms: ClassVar[dict] = {
        None: ["scheme_euler_explicit_ale", "schema_euler_explicite_ale"],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Jones_launder(Modele_fonction_bas_reynolds_base):
    r"""
    Model described in ' Jones, W. P. and Launder, B. E. (1972), The prediction of
    laminarization with a two-equation model of turbulence, Int. J. of Heat and Mass transfer,
    Vol. 15, pp. 301-314.'
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Extraire_surface_ale(Interprete):
    r"""
    Extraire_surface_ALE in order to extract a surface on a mobile boundary (with ALE
    desciption).

    Keyword to specify that the extract surface is done on a mobile domain. The surface mesh
    is defined by one or two conditions. The first condition is about elements with
    Condition_elements. For example: Condition_elements x*x+y*y+z*z<1

    Will define a surface mesh with external faces of the mesh elements inside the sphere of
    radius 1 located at (0,0,0). The second condition Condition_faces is useful to give a
    restriction.

    By default, the faces from the boundaries are not added to the surface mesh excepted if
    option avec_les_bords is given (all the boundaries are added), or if the option
    avec_certains_bords is used to add only some boundaries.
    """

    domaine: str = Field(description=r"""Domain in which faces are saved""", default="")
    probleme: str = Field(
        description=r"""Problem from which faces should be extracted""", default=""
    )
    condition_elements: Optional[str] = Field(description=r"""not_set""", default=None)
    condition_faces: Optional[str] = Field(description=r"""not_set""", default=None)
    avec_les_bords: Optional[bool] = Field(description=r"""not_set""", default=None)
    avec_certains_bords: Optional[List[str]] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "domaine": [],
        "probleme": [],
        "condition_elements": [],
        "condition_faces": [],
        "avec_les_bords": [],
        "avec_certains_bords": [],
    }


################################################################


class Convection_diffusion_phase_field(Convection_diffusion_concentration):
    r"""
    Cahn-Hilliard equation of the Phase Field problem. The unknown of this equation is the
    concentration C.
    """

    mu_1: Optional[float] = Field(
        description=r"""Dynamic viscosity of the first phase.""", default=None
    )
    mu_2: Optional[float] = Field(
        description=r"""Dynamic viscosity of the second phase.""", default=None
    )
    rho_1: Optional[float] = Field(
        description=r"""Density of the first phase.""", default=None
    )
    rho_2: Optional[float] = Field(
        description=r"""Density of the second phase.""", default=None
    )
    potentiel_chimique_generalise: Literal[
        "avec_energie_cinetique", "sans_energie_cinetique"
    ] = Field(
        description=r"""To define (chaine set to avec_energie_cinetique) or not (chaine set to sans_energie_cinetique) if the Cahn-Hilliard equation contains the cinetic energy term.""",
        default="avec_energie_cinetique",
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "mu_1": [],
        "mu_2": [],
        "rho_1": [],
        "rho_2": [],
        "potentiel_chimique_generalise": [],
        "nom_inconnue": [],
        "alias": [],
        "masse_molaire": [],
        "is_multi_scalar": ["is_multi_scalar_diffusion"],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Pb_phase_field(Pb_base):
    r"""
    Problem to solve local instantaneous incompressible-two-phase-flows. Complete description
    of the Phase Field model for incompressible and immiscible fluids can be found into this
    PDF: TRUST_ROOT/doc/TRUST/phase_field_non_miscible_manuel.pdf
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituents.""", default=None
    )
    navier_stokes_phase_field: Optional[Navier_stokes_phase_field] = Field(
        description=r"""Navier Stokes equation for the Phase Field problem.""",
        default=None,
    )
    convection_diffusion_phase_field: Optional[Convection_diffusion_phase_field] = (
        Field(
            description=r"""Cahn-Hilliard equation of the Phase Field problem. The unknown of this equation is the concentration C.""",
            default=None,
        )
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "constituant": [],
        "navier_stokes_phase_field": [],
        "convection_diffusion_phase_field": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Lam_bremhorst(Modele_fonction_bas_reynolds_base):
    r"""
    Model described in ' C.K.G.Lam and K.Bremhorst, A modified form of the k- epsilon model
    for predicting wall turbulence, ASME J. Fluids Engng., Vol.103, p456, (1981)'. Only in
    VEF.
    """

    fichier_distance_paroi: Optional[str] = Field(
        description=r"""refer to distance_paroi keyword""", default=None
    )
    reynolds_stress_isotrope: Optional[int] = Field(
        description=r"""keyword for isotropic Reynolds stress""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fichier_distance_paroi": [],
        "reynolds_stress_isotrope": [],
    }


################################################################


class Navier_stokes_standard_sensibility(Navier_stokes_standard):
    r"""
    Resolution of Navier-Stokes sensitivity problem
    """

    state: Optional[Bloc_lecture] = Field(
        description=r"""Block to indicate the state problem. Between the braces, you must specify the key word 'pb_champ_evaluateur' then the name of the state problem and the velocity unknown  Example: state { pb_champ_evaluateur pb_state velocity }""",
        default=None,
    )
    uncertain_variable: Optional[Bloc_lecture] = Field(
        description=r"""Block to indicate the name of the uncertain variable. Between the braces, you must specify the name of the unknown variable. Choice between velocity and mu.  Example: uncertain_variable { velocity }""",
        default=None,
    )
    polynomial_chaos: Optional[float] = Field(
        description=r"""It is the method that we will use to study the sensitivity of the Navier Stokes equation:  if poly_chaos=0, the sensitivity will be treated by the standard sentivity method. If different than 0, it will be treated by the polynomial chaos method""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "state": [],
        "uncertain_variable": [],
        "polynomial_chaos": [],
        "correction_matrice_projection_initiale": [],
        "correction_calcul_pression_initiale": [],
        "correction_vitesse_projection_initiale": [],
        "correction_matrice_pression": [],
        "correction_vitesse_modifie": [],
        "gradient_pression_qdm_modifie": [],
        "correction_pression_modifie": [],
        "postraiter_gradient_pression_sans_masse": [],
        "solveur_pression": [],
        "dt_projection": [],
        "traitement_particulier": [],
        "seuil_divu": [],
        "solveur_bar": [],
        "projection_initiale": [],
        "methode_calcul_pression_initiale": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class K_epsilon_bicephale(Mod_turb_hyd_rans_bicephale):
    r"""
    Turbulence model (k-eps) en formalisation bicephale.
    """

    transport_k: Optional[str] = Field(
        description=r"""Keyword to define the realisable (k) transportation equation.""",
        default=None,
    )
    transport_epsilon: Optional[str] = Field(
        description=r"""Keyword to define the realisable (eps) transportation equation.""",
        default=None,
    )
    modele_fonc_bas_reynolds: Optional[Modele_fonc_realisable_base] = Field(
        description=r"""This keyword is used to set the model used""", default=None
    )
    cmu: Optional[float] = Field(
        description=r"""Keyword to modify the Cmu constant of k-eps model : Nut=Cmu*k*k/eps Default value is 0.09""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "transport_k": [],
        "transport_epsilon": [],
        "modele_fonc_bas_reynolds": [],
        "cmu": [],
        "eps_min": [],
        "eps_max": [],
        "prandtl_k": [],
        "prandtl_eps": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Frontiere_ouverte_vitesse_imposee_ale(Dirichlet):
    r"""
    Class for velocity boundary condition on a mobile boundary (ALE framework).

    The imposed velocity field is vectorial of type Ch_front_input_ALE, Champ_front_ALE or
    Champ_front_ALE_Beam.

    Example: frontiere_ouverte_vitesse_imposee_ALE Champ_front_ALE 2 0.5*cos(0.5*t) 0.0
    """

    ch: Front_field_base = Field(
        description=r"""Boundary field type.""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "ch": []}


################################################################


class Correction_lubchenko(Source_base):
    r"""
    not_set
    """

    beta_lift: Optional[float] = Field(description=r"""not_set""", default=None)
    beta_disp: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {None: [], "beta_lift": [], "beta_disp": []}


################################################################


class Solver_moving_mesh_ale(Interprete):
    r"""
    Solver used to solve the system giving the mesh velocity for the ALE (Arbitrary
    Lagrangian-Eulerian) framework.
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    bloc: Bloc_lecture = Field(
        description=r"""Example: { PETSC GCP { precond ssor { omega 1.5 } seuil 1e-7 impr } }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Source_qdm_phase_field(Source_base):
    r"""
    Keyword to define the capillary force into the Navier Stokes equation for the Phase Field
    problem.
    """

    forme_du_terme_source: int = Field(
        description=r"""Kind of the source term (1, 2, 3 or 4).""", default=0
    )
    _synonyms: ClassVar[dict] = {None: [], "forme_du_terme_source": []}


################################################################


class Source_dissipation_echelle_temp_taux_diss_turb(Source_base):
    r"""
    Source term which corresponds to the dissipation source term that appears in the transport
    equation for tau (in the k-tau turbulence model)
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Implicit_steady(Implicite):
    r"""
    this is the implicit solver using a dual time step. Remark: this solver can be used only
    with the Implicit_Euler_Steady_Scheme time scheme.
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "seuil_convergence_implicite": [],
        "nb_corrections_max": [],
        "seuil_convergence_solveur": [],
        "seuil_generation_solveur": [],
        "seuil_verification_solveur": [],
        "seuil_test_preliminaire_solveur": [],
        "solveur": [],
        "no_qdm": [],
        "nb_it_max": [],
        "controle_residu": [],
    }


################################################################


class Shih_zhu_lumley(Modele_fonc_realisable_base):
    r"""
    Functions necessary to Realizable K-Epsilon Turbulence Model in VEF
    """

    a0: Optional[float] = Field(
        description=r"""value of parameter A0 in U* formula""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "a0": []}


################################################################


class Easm_baglietto(Lam_bremhorst):
    r"""
    Model described in ' E. Baglietto and H. Ninokata , A turbulence model study for
    simulating flow inside tight lattice rod bundles, Nuclear Engineering and Design, 773--784
    (235), 2005. '
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fichier_distance_paroi": [],
        "reynolds_stress_isotrope": [],
    }


################################################################


class Standard_keps(Lam_bremhorst):
    r"""
    Model described in ' E. Baglietto , CFD and DNS methodologies development for fuel bundle
    simulaions, Nuclear Engineering and Design, 1503--1510 (236), 2006. '
    """

    _synonyms: ClassVar[dict] = {
        None: [],
        "fichier_distance_paroi": [],
        "reynolds_stress_isotrope": [],
    }


################################################################


class Domaine_ale(Domaine):
    r"""
    Domain with nodes at the interior of the domain which are displaced in an arbitrarily
    prescribed way thanks to ALE (Arbitrary Lagrangian-Eulerian) description.

    Keyword to specify that the domain is mobile following the displacement of some of its
    boundaries.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Modele_shih_zhu_lumley_vdf(Modele_fonc_realisable_base):
    r"""
    Functions necessary to Realizable K-Epsilon Turbulence Model in VDF
    """

    a0: Optional[float] = Field(
        description=r"""value of parameter A0 in U* formula""", default=None
    )
    _synonyms: ClassVar[dict] = {None: [], "a0": []}


################################################################


class Boundary_field_keps_from_ud(Front_field_base):
    r"""
    To specify a K-Eps inlet field with hydraulic diameter, speed, and turbulence intensity
    (VDF only)
    """

    u: Front_field_base = Field(
        description=r"""U 0 Initial velocity magnitude""",
        default_factory=lambda: eval("Front_field_base()"),
    )
    d: float = Field(description=r"""Hydraulic diameter""", default=0.0)
    i: float = Field(description=r"""Turbulence intensity [%]""", default=0.0)
    _synonyms: ClassVar[dict] = {None: [], "u": [], "d": [], "i": []}


################################################################


class Tenseur_reynolds_externe(Source_base):
    r"""
    Use a neural network to estimate the values of the Reynolds tensor. The structure of the
    neural networks is stored in a file located in the share/reseaux_neurones directory.
    """

    nom_fichier: str = Field(description=r"""The base name of the file.""", default="")
    _synonyms: ClassVar[dict] = {None: [], "nom_fichier": []}


################################################################


class K_epsilon(Mod_turb_hyd_rans_keps):
    r"""
    Turbulence model (k-eps).
    """

    transport_k_epsilon: Optional[Transport_k_epsilon] = Field(
        description=r"""Keyword to define the (k-eps) transportation equation.""",
        default=None,
    )
    modele_fonc_bas_reynolds: Optional[Modele_fonction_bas_reynolds_base] = Field(
        description=r"""This keyword is used to set the bas Reynolds model used.""",
        default=None,
    )
    cmu: Optional[float] = Field(
        description=r"""Keyword to modify the Cmu constant of k-eps model : Nut=Cmu*k*k/eps Default value is 0.09""",
        default=None,
    )
    prandtl_k: Optional[float] = Field(
        description=r"""Keyword to change the Prk value (default 1.0).""", default=None
    )
    prandtl_eps: Optional[float] = Field(
        description=r"""Keyword to change the Pre value (default 1.3).""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "transport_k_epsilon": [],
        "modele_fonc_bas_reynolds": [],
        "cmu": [],
        "prandtl_k": [],
        "prandtl_eps": [],
        "eps_min": [],
        "eps_max": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Systeme_naire_deriv(Objet_lecture):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Source_con_phase_field(Source_base):
    r"""
    Keyword to define the source term of the Cahn-Hilliard equation.
    """

    systeme_naire: Optional[Systeme_naire_deriv] = Field(
        description=r"""not_set""", default=None
    )
    temps_d_affichage: int = Field(
        description=r"""Time during the caracteristics of the problem are shown before calculation.""",
        default=0,
    )
    moyenne_de_kappa: str = Field(
        description=r"""To define how mobility kappa is calculated on faces of the mesh according to cell-centered values (chaine is arithmetique/harmonique/geometrique).""",
        default="",
    )
    multiplicateur_de_kappa: float = Field(
        description=r"""To define the parameter of the mobility expression when mobility depends on C.""",
        default=0.0,
    )
    couplage_ns_ch: str = Field(
        description=r"""Evaluating time choosen for the term source calculation into the Navier Stokes equation (chaine is mutilde(n+1/2)/mutilde(n), in order to be conservative, the first choice seems better).""",
        default="",
    )
    implicitation_ch: Literal["oui", "non"] = Field(
        description=r"""To define if the Cahn-Hilliard will be solved using a implicit algorithm or not.""",
        default="oui",
    )
    gmres_non_lineaire: Literal["oui", "non"] = Field(
        description=r"""To define the algorithm to solve Cahn-Hilliard equation (oui: Newton-Krylov method, non: fixed point method).""",
        default="oui",
    )
    seuil_cv_iterations_ptfixe: float = Field(
        description=r"""Convergence threshold (an option of the fixed point method).""",
        default=0.0,
    )
    seuil_residu_ptfixe: float = Field(
        description=r"""Threshold for the matrix inversion used in the method (an option of the fixed point method).""",
        default=0.0,
    )
    seuil_residu_gmresnl: float = Field(
        description=r"""Convergence threshold (an option of the Newton-Krylov method).""",
        default=0.0,
    )
    dimension_espace_de_krylov: int = Field(
        description=r"""Vector numbers used in the method (an option of the Newton-Krylov method).""",
        default=0,
    )
    nb_iterations_gmresnl: int = Field(
        description=r"""Maximal iteration (an option of the Newton-Krylov method).""",
        default=0,
    )
    residu_min_gmresnl: float = Field(
        description=r"""Minimal convergence threshold (an option of the Newton-Krylov method).""",
        default=0.0,
    )
    residu_max_gmresnl: float = Field(
        description=r"""Maximal convergence threshold (an option of the Newton-Krylov method).""",
        default=0.0,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "systeme_naire": [],
        "temps_d_affichage": [],
        "moyenne_de_kappa": [],
        "multiplicateur_de_kappa": [],
        "couplage_ns_ch": [],
        "implicitation_ch": [],
        "gmres_non_lineaire": [],
        "seuil_cv_iterations_ptfixe": [],
        "seuil_residu_ptfixe": [],
        "seuil_residu_gmresnl": [],
        "dimension_espace_de_krylov": [],
        "nb_iterations_gmresnl": [],
        "residu_min_gmresnl": [],
        "residu_max_gmresnl": [],
    }


################################################################


class Bloc_kappa_variable(Objet_lecture):
    r"""
    if the parameter of the mobility, kappa, depends on C
    """

    expr: Bloc_lecture = Field(
        description=r"""choice for kappa_variable""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "expr": []}


################################################################


class Bloc_potentiel_chim(Objet_lecture):
    r"""
    if the chemical potential function is an univariate function
    """

    expr: Bloc_lecture = Field(
        description=r"""choice for potentiel_chimique""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "expr": []}


################################################################


class Systeme_naire_non(Systeme_naire_deriv):
    r"""
    not_set
    """

    alpha: float = Field(
        description=r"""Internal capillary coefficient alfa.""", default=0.0
    )
    beta: float = Field(description=r"""Parameter beta of the model.""", default=0.0)
    kappa: float = Field(description=r"""Mobility coefficient kappa0.""", default=0.0)
    kappa_variable: Bloc_kappa_variable = Field(
        description=r"""To define a mobility which depends on concentration C.""",
        default_factory=lambda: eval("Bloc_kappa_variable()"),
    )
    potentiel_chimique: Optional[Bloc_potentiel_chim] = Field(
        description=r"""chemical potential function""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["non"],
        "alpha": [],
        "beta": [],
        "kappa": [],
        "kappa_variable": [],
        "potentiel_chimique": [],
    }


################################################################


class Pb_hydraulique_sensibility(Pb_base):
    r"""
    Resolution of hydraulic sensibility problems
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    navier_stokes_standard_sensibility: Navier_stokes_standard_sensibility = Field(
        description=r"""Navier-Stokes sensibility equations""",
        default_factory=lambda: eval("Navier_stokes_standard_sensibility()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "navier_stokes_standard_sensibility": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Newmarktimescheme_deriv(Objet_lecture):
    r"""
    Solve the beam dynamics. Selection of time integration scheme.
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Bloc_poutre(Objet_lecture):
    r"""
    Read poutre bloc
    """

    nb_modes: int = Field(description=r"""Number of modes""", default=0)
    direction: int = Field(description=r"""x=0, y=1, z=2""", default=0)
    newmarktimescheme: Newmarktimescheme_deriv = Field(
        description=r"""Solve the beam dynamics. Time integration scheme: choice between MA (Newmark mean acceleration), FD (Newmark finite differences), and HHT alpha (Hilber-Hughes-Taylor, alpha usually -0.1 )""",
        default_factory=lambda: eval("Newmarktimescheme_deriv()"),
    )
    mass_and_stiffness_file_name: str = Field(
        description=r"""Name of the file containing the diagonal modal mass, stiffness, and damping matrices.""",
        default="",
    )
    absc_file_name: str = Field(
        description=r"""Name of the file containing the coordinates of the Beam""",
        default="",
    )
    modal_deformation_file_name: List[str] = Field(
        description=r"""Name of the file containing the modal deformation of the Beam (mandatory if different from 0. 0. 0.)""",
        default_factory=list,
    )
    young_module: Optional[float] = Field(description=r"""Young Module""", default=None)
    rho_beam: Optional[float] = Field(description=r"""Beam density""", default=None)
    basecentercoordinates: Optional[Annotated[List[float], "size_is_dim"]] = Field(
        description=r"""position of the base center coordinates on the Beam""",
        default=None,
    )
    ci_file_name: Optional[str] = Field(
        description=r"""Name of the file containing the initial condition of the Beam""",
        default=None,
    )
    restart_file_name: Optional[str] = Field(
        description=r"""SaveBeamForRestart.txt file to restart the calculation""",
        default=None,
    )
    output_position_1d: Optional[List[float]] = Field(
        description=r"""nb_points position Post-traitement of specific points on the Beam""",
        default=None,
    )
    output_position_3d: Optional[Annotated[List[Un_point], "Listpoints"]] = Field(
        description=r"""Points.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "nb_modes": ["n"],
        "direction": ["dir"],
        "newmarktimescheme": [],
        "mass_and_stiffness_file_name": [],
        "absc_file_name": [],
        "modal_deformation_file_name": [],
        "young_module": ["young"],
        "rho_beam": ["rho"],
        "basecentercoordinates": ["pos_center"],
        "ci_file_name": [],
        "restart_file_name": [],
        "output_position_1d": ["pt1d"],
        "output_position_3d": ["pt3d"],
    }


################################################################


class Bloc_lecture_beam_model(Objet_lecture):
    r"""
    bloc
    """

    aco: Literal["{"] = Field(description=r"""Opening curly bracket.""", default="{")
    nb_beam: Literal["nb_beam"] = Field(
        description=r"""Keyword to specify the number of beams""", default="nb_beam"
    )
    nb_beam_val: int = Field(description=r"""Number of beams""", default=0)
    name: Literal["name"] = Field(
        description=r"""keyword to specify the Name of the beam (the name must match with the name of the edge in the fluid domain)""",
        default="name",
    )
    name_of_beam: str = Field(
        description=r"""keyword to specify the Name of the beam (the name must match with the name of the edge in the fluid domain)""",
        default="",
    )
    bloc: Bloc_poutre = Field(
        description=r"""not_set""", default_factory=lambda: eval("Bloc_poutre()")
    )
    name2: Optional[Literal["name"]] = Field(
        description=r"""keyword to specify the Name of the beam (the name must match with the name of the edge in the fluid domain)""",
        default=None,
    )
    name_of_beam2: Optional[str] = Field(
        description=r"""keyword to specify the Name of the beam (the name must match with the name of the edge in the fluid domain)""",
        default=None,
    )
    bloc2: Optional[Bloc_poutre] = Field(description=r"""not_set""", default=None)
    acof: Literal["}"] = Field(description=r"""Closing curly bracket.""", default="}")
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "aco": [],
        "nb_beam": [],
        "nb_beam_val": [],
        "name": ["beamname"],
        "name_of_beam": [],
        "bloc": [],
        "name2": ["beamname2"],
        "name_of_beam2": [],
        "bloc2": [],
        "acof": [],
    }


################################################################


class Beam_model(Interprete):
    r"""
    Reduced mechanical model: a beam model. Resolution based on a modal analysis. Temporal
    discretization: Newmark or Hilber-Hughes-Taylor (HHT)
    """

    dom: str = Field(description=r"""domain name""", default="")
    bloc: Bloc_lecture_beam_model = Field(
        description=r"""not_set""",
        default_factory=lambda: eval("Bloc_lecture_beam_model()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Newmarktimescheme_hhr(Newmarktimescheme_deriv):
    r"""
    HHT alpha (Hilber-Hughes-Taylor, alpha usually -0.1 ) time integration scheme.
    """

    alpha: Optional[float] = Field(
        description=r"""usually, alpha is set to -0.1""", default=None
    )
    _synonyms: ClassVar[dict] = {None: ["hht"], "alpha": []}


################################################################


class Newmarktimescheme_ma(Newmarktimescheme_deriv):
    r"""
    MA (Newmark mean acceleration) time integration scheme.
    """

    _synonyms: ClassVar[dict] = {None: ["ma"]}


################################################################


class Newmarktimescheme_fd(Newmarktimescheme_deriv):
    r"""
    FD (Newmark finite differences) time integration scheme.
    """

    _synonyms: ClassVar[dict] = {None: ["fd"]}


################################################################


class Source_dissipation_hzdr(Source_base):
    r"""
    Additional source terms in the turbulent dissipation (omega) equation to model the
    fluctuations induced by bubbles.
    """

    constante_gravitation: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    c_k: Optional[float] = Field(description=r"""not_set""", default=None)
    c_epsilon: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "constante_gravitation": [],
        "c_k": [],
        "c_epsilon": [],
    }


################################################################


class Type_diffusion_turbulente_multiphase_k_tau(
    Type_diffusion_turbulente_multiphase_deriv
):
    r"""
    not_set
    """

    limiter: Optional[str] = Field(description=r"""not_set""", default=None)
    sigma: Optional[float] = Field(description=r"""not_set""", default=None)
    beta_k: Optional[float] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: ["k_tau"],
        "limiter": ["limiteur"],
        "sigma": [],
        "beta_k": [],
    }


################################################################


class Paroi_frottante_simple(Condlim_base):
    r"""
    Adaptive wall-law boundary condition for velocity
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Champ_front_ale(Front_field_base):
    r"""
    Class to define a boundary condition on a moving boundary of a mesh (only for the
    Arbitrary Lagrangian-Eulerian framework ).
    """

    val: List[str] = Field(
        description=r""" Example: 2 -y*0.01 x*0.01""", default_factory=list
    )
    _synonyms: ClassVar[dict] = {None: [], "val": []}


################################################################


class Convection_diffusion_temperature_sensibility(Convection_diffusion_temperature):
    r"""
    Energy sensitivity equation (temperature diffusion convection)
    """

    convection_sensibility: Optional[Convection_deriv] = Field(
        description=r"""Choice between: amont and muscl  Example: convection { Sensibility { amont } }""",
        default=None,
    )
    velocity_state: Optional[Bloc_lecture] = Field(
        description=r"""Block to indicate the state problem. Between the braces, you must specify the key word 'pb_champ_evaluateur' then the name of the state problem and the velocity unknown  Example: velocity_state { pb_champ_evaluateur pb_state velocity }""",
        default=None,
    )
    temperature_state: Optional[Bloc_lecture] = Field(
        description=r"""Block to indicate the state problem. Between the braces, you must specify the key word 'pb_champ_evaluateur' then the name of the state problem and the temperature unknown  Example: velocity_state { pb_champ_evaluateur pb_state temperature }""",
        default=None,
    )
    uncertain_variable: Optional[Bloc_lecture] = Field(
        description=r"""Block to indicate the name of the uncertain variable. Between the braces, you must specify the name of the unknown variable (choice between: temperature, beta_th, boussinesq_temperature, Cp and lambda .  Example: uncertain_variable { temperature }""",
        default=None,
    )
    polynomial_chaos: Optional[float] = Field(
        description=r"""It is the method that we will use to study the sensitivity of the""",
        default=None,
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "convection_sensibility": ["sensibility"],
        "velocity_state": [],
        "temperature_state": [],
        "uncertain_variable": [],
        "polynomial_chaos": [],
        "penalisation_l2_ftd": [],
        "disable_equation_residual": [],
        "convection": [],
        "diffusion": [],
        "conditions_limites": ["boundary_conditions"],
        "conditions_initiales": ["initial_conditions"],
        "sources": [],
        "ecrire_fichier_xyz_valeur": [],
        "parametre_equation": [],
        "equation_non_resolue": [],
        "renommer_equation": ["rename_equation"],
    }


################################################################


class Fluid_diph_lu(Objet_lecture):
    r"""
    Single fluid to be read.
    """

    fluid_name: str = Field(
        description=r"""Name of the fluid which is part of the diphasic fluid.""",
        default="",
    )
    single_fld: Fluide_incompressible = Field(
        description=r"""Definition of the single fluid part of a multiphasic fluid.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    _synonyms: ClassVar[dict] = {None: ["nul"], "fluid_name": [], "single_fld": []}


################################################################


class Fluide_diphasique(Milieu_base):
    r"""
    fluid_diph_lu 0 Two-phase fluid.
    """

    sigma: Champ_don_base = Field(
        description=r"""surfacic tension (J/m2)""",
        default_factory=lambda: eval("Champ_don_base()"),
    )
    fluide0: Fluid_diph_lu = Field(
        description=r"""first phase fluid""",
        default_factory=lambda: eval("Fluid_diph_lu()"),
    )
    fluide1: Fluid_diph_lu = Field(
        description=r"""second phase fluid""",
        default_factory=lambda: eval("Fluid_diph_lu()"),
    )
    chaleur_latente: Optional[Champ_don_base] = Field(
        description=r"""phase changement enthalpy h(phase1_) - h(phase0_) (J/kg/K)""",
        default=None,
    )
    formule_mu: Optional[str] = Field(
        description=r"""(into=[standard,arithmetic,harmonic]) formula used to calculate average""",
        default=None,
    )
    gravite: Optional[Field_base] = Field(description=r"""not_set""", default=None)
    _synonyms: ClassVar[dict] = {
        None: [],
        "sigma": [],
        "fluide0": ["phase0"],
        "fluide1": ["phase1"],
        "chaleur_latente": [],
        "formule_mu": [],
        "gravite": [],
        "porosites_champ": [],
        "diametre_hyd_champ": [],
        "porosites": [],
        "rho": [],
        "lambda_": ["lambda_u", "lambda"],
        "cp": [],
    }


################################################################


class Remaillage_ft_ijk(Interprete):
    r"""
    not_set
    """

    pas_remaillage: Optional[float] = Field(description=r"""not_set""", default=None)
    nb_iter_barycentrage: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    relax_barycentrage: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    critere_arete: Optional[float] = Field(description=r"""not_set""", default=None)
    seuil_dvolume_residuel: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    nb_iter_correction_volume: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    nb_iter_remaillage: Optional[int] = Field(description=r"""not_set""", default=None)
    facteur_longueur_ideale: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    equilateral: Optional[int] = Field(description=r"""not_set""", default=None)
    lissage_courbure_coeff: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    lissage_courbure_iterations_systematique: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    lissage_courbure_iterations_si_remaillage: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "pas_remaillage": [],
        "nb_iter_barycentrage": [],
        "relax_barycentrage": [],
        "critere_arete": [],
        "seuil_dvolume_residuel": [],
        "nb_iter_correction_volume": [],
        "nb_iter_remaillage": [],
        "facteur_longueur_ideale": [],
        "equilateral": [],
        "lissage_courbure_coeff": [],
        "lissage_courbure_iterations_systematique": [],
        "lissage_courbure_iterations_si_remaillage": [],
    }


################################################################


class Convection_rt(Convection_deriv):
    r"""
    Keyword to use RT projection for P1NCP0RT discretization
    """

    _synonyms: ClassVar[dict] = {None: ["rt"]}


################################################################


class Imposer_vit_bords_ale(Interprete):
    r"""
    For the Arbitrary Lagrangian-Eulerian framework: block to indicate the number of mobile
    boundaries of the domain and specify the speed that must be imposed on them.
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    bloc: Bloc_lecture = Field(
        description=r"""between the braces, you must specify the numbers of the mobile borders of the domain then list these mobile borders and indicate the speed which must be imposed on them  Example: Imposer_vit_bords_ALE dom_name { 1 boundary_name Champ_front_ALE 2 -(y-0.1)*0.01 (x-0.1)*0.01 }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Diffusion_supplementaire_lin_echelle_temp_turb(Source_base):
    r"""
    not_set
    """

    _synonyms: ClassVar[dict] = {None: ["diffusion_supplementaire_echelle_temp_turb"]}


################################################################


class Schema_phase_field(Schema_temps_base):
    r"""
    Keyword for the only available Scheme for time discretization of the Phase Field problem.
    """

    schema_ch: Optional[Schema_temps_base] = Field(
        description=r"""Time scheme for the Cahn-Hilliard equation.""", default=None
    )
    schema_ns: Optional[Schema_temps_base] = Field(
        description=r"""Time scheme for the Navier-Stokes equation.""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "schema_ch": [],
        "schema_ns": [],
        "tinit": [],
        "tmax": [],
        "tcpumax": [],
        "dt_min": [],
        "dt_max": [],
        "dt_sauv": [],
        "dt_impr": [],
        "facsec": [],
        "seuil_statio": [],
        "residuals": [],
        "diffusion_implicite": [],
        "seuil_diffusion_implicite": [],
        "impr_diffusion_implicite": [],
        "impr_extremums": [],
        "no_error_if_not_converged_diffusion_implicite": [],
        "no_conv_subiteration_diffusion_implicite": [],
        "dt_start": [],
        "nb_pas_dt_max": [],
        "niter_max_diffusion_implicite": [],
        "precision_impr": [],
        "periode_sauvegarde_securite_en_heures": [],
        "no_check_disk_space": [],
        "disable_progress": [],
        "disable_dt_ev": [],
        "gnuplot_header": [],
    }


################################################################


class Ale_neumann_bc_for_grid_problem(Interprete):
    r"""
    block to indicates the names of the boundary with Neumann BC for the grid problem. By
    default, in the ALE grid problem, we impose a homogeneous Dirichelt-type BC on the fix
    boundary. This option allows you to impose also Neumann-type BCs on certain boundary.
    """

    dom: str = Field(description=r"""Name of domain.""", default="")
    bloc: Bloc_lecture = Field(
        description=r"""between the braces, you must specify the numbers of the mobile borders then list these mobile borders.  Example: ALE_Neumann_BC_for_grid_problem dom_name { 1 boundary_name }""",
        default_factory=lambda: eval("Bloc_lecture()"),
    )
    _synonyms: ClassVar[dict] = {None: [], "dom": [], "bloc": []}


################################################################


class Interfaces(Interprete):
    r"""
    not_set
    """

    fichier_reprise_interface: str = Field(description=r"""not_set""", default="")
    timestep_reprise_interface: Optional[int] = Field(
        description=r"""not_set""", default=None
    )
    lata_meshname: Optional[str] = Field(description=r"""not_set""", default=None)
    remaillage_ft_ijk: Optional[Remaillage_ft_ijk] = Field(
        description=r"""not_set""", default=None
    )
    no_octree_method: Optional[int] = Field(
        description=r"""if the bubbles repel each other, what method should be used to compute relative velocities? Octree method by default, otherwise we used the IJK discretization""",
        default=None,
    )
    compute_distance_autres_interfaces: Optional[bool] = Field(
        description=r"""not_set""", default=None
    )
    terme_gravite: Optional[Literal["rho_g", "grad_i"]] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["nul"],
        "fichier_reprise_interface": [],
        "timestep_reprise_interface": [],
        "lata_meshname": [],
        "remaillage_ft_ijk": [],
        "no_octree_method": [],
        "compute_distance_autres_interfaces": [],
        "terme_gravite": [],
    }


################################################################


class Cond_lim_k_complique_transition_flux_nul_demi(Condlim_base):
    r"""
    Adaptive wall law boundary condition for turbulent kinetic energy
    """

    _synonyms: ClassVar[dict] = {None: []}


################################################################


class Echange_contact_vdf_ft_disc(Condlim_base):
    r"""
    echange_conatct_vdf en prescisant la phase
    """

    autre_probleme: str = Field(description=r"""name of other problem""", default="")
    autre_bord: str = Field(description=r"""name of other boundary""", default="")
    autre_champ_temperature: str = Field(
        description=r"""name of other field""", default=""
    )
    nom_mon_indicatrice: str = Field(description=r"""name of indicatrice""", default="")
    phase: int = Field(description=r"""phase""", default=0)
    _synonyms: ClassVar[dict] = {
        None: [],
        "autre_probleme": [],
        "autre_bord": [],
        "autre_champ_temperature": [],
        "nom_mon_indicatrice": [],
        "phase": [],
    }


################################################################


class Pb_thermohydraulique_sensibility(Pb_thermohydraulique):
    r"""
    Resolution of Resolution of thermohydraulic sensitivity problem
    """

    fluide_incompressible: Fluide_incompressible = Field(
        description=r"""The fluid medium associated with the problem.""",
        default_factory=lambda: eval("Fluide_incompressible()"),
    )
    convection_diffusion_temperature_sensibility: Convection_diffusion_temperature_sensibility = Field(
        description=r"""Convection diffusion temperature sensitivity equation""",
        default_factory=lambda: eval("Convection_diffusion_temperature_sensibility()"),
    )
    navier_stokes_standard_sensibility: Navier_stokes_standard_sensibility = Field(
        description=r"""Navier Stokes sensitivity equation""",
        default_factory=lambda: eval("Navier_stokes_standard_sensibility()"),
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "fluide_incompressible": [],
        "convection_diffusion_temperature_sensibility": [
            "convection_diffusion_temperature"
        ],
        "navier_stokes_standard_sensibility": [],
        "fluide_ostwald": [],
        "fluide_sodium_liquide": [],
        "fluide_sodium_gaz": [],
        "correlations": [],
        "navier_stokes_standard": [],
        "convection_diffusion_temperature": [],
        "milieu": [],
        "constituant": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class K_eps_realisable(Mod_turb_hyd_rans):
    r"""
    Realizable K-Epsilon Turbulence Model.
    """

    transport_k_epsilon_realisable: Optional[str] = Field(
        description=r"""Keyword to define the realisable (k-eps) transportation equation.""",
        default=None,
    )
    modele_fonc_realisable: Optional[Modele_fonc_realisable_base] = Field(
        description=r"""This keyword is used to set the model used""", default=None
    )
    prandtl_k: float = Field(
        description=r"""Keyword to change the Prk value (default 1.0).""", default=0.0
    )
    prandtl_eps: float = Field(
        description=r"""Keyword to change the Pre value (default 1.3)""", default=0.0
    )
    _synonyms: ClassVar[dict] = {
        None: ["k_epsilon_realisable"],
        "transport_k_epsilon_realisable": [],
        "modele_fonc_realisable": [],
        "prandtl_k": [],
        "prandtl_eps": [],
        "k_min": [],
        "quiet": [],
        "turbulence_paroi": [],
        "dt_impr_ustar": [],
        "dt_impr_ustar_mean_only": [],
        "nut_max": [],
        "correction_visco_turb_pour_controle_pas_de_temps": [],
        "correction_visco_turb_pour_controle_pas_de_temps_parametre": [],
    }


################################################################


class Listdeuxmots_acc(Listobj):
    r"""
    List of groups of two words (with curly brackets).
    """

    _synonyms: ClassVar[dict] = {None: ["nul"]}


################################################################


class Pb_fronttracking_disc(Problem_read_generic):
    r"""
    The generic Front-Tracking problem in the discontinuous version. It differs from the rest
    of the TRUST code : The problem does not state the number of equations that are enclosed
    in the problem. Two equations are compulsory : a momentum balance equation (alias Navier-
    Stokes equation) and an interface tracking equation. The list of equations to be solved is
    declared in the beginning of the data file. Another difference with more classical TRUST
    data file, lies in the fluids definition. The two-phase fluid (Fluide_Diphasique) is made
    with two usual single-phase fluids (Fluide_Incompressible). As the list of equations to be
    solved in the generic Front-Tracking problem is declared in the data file and not pre-
    defined in the structure of the problem, each equation has to be distinctively associated
    with the problem with the Associer keyword.
    """

    solved_equations: Annotated[List[Deuxmots], "Listdeuxmots_acc"] = Field(
        description=r"""List of groups of two words (with curly brackets).""",
        default_factory=list,
    )
    fluide_incompressible: Optional[Fluide_incompressible] = Field(
        description=r"""The fluid medium associated with the problem.""", default=None
    )
    fluide_diphasique: Optional[Fluide_diphasique] = Field(
        description=r"""The diphasic fluid medium associated with the problem.""",
        default=None,
    )
    constituant: Optional[Constituant] = Field(
        description=r"""Constituent.""", default=None
    )
    triple_line_model_ft_disc: Optional[Triple_line_model_ft_disc] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: ["probleme_ft_disc_gen"],
        "solved_equations": [],
        "fluide_incompressible": [],
        "fluide_diphasique": [],
        "constituant": [],
        "triple_line_model_ft_disc": [],
        "milieu": [],
        "postraitement": ["post_processing"],
        "postraitements": ["post_processings"],
        "liste_de_postraitements": [],
        "liste_postraitements": [],
        "sauvegarde": [],
        "sauvegarde_simple": [],
        "reprise": [],
        "resume_last_time": [],
    }


################################################################


class Debogft(Interprete):
    r"""
    not_set
    """

    mode: Optional[Literal["disabled", "write_pass", "check_pass"]] = Field(
        description=r"""not_set""", default=None
    )
    filename: Optional[str] = Field(description=r"""not_set""", default=None)
    seuil_absolu: Optional[float] = Field(description=r"""not_set""", default=None)
    seuil_relatif: Optional[float] = Field(description=r"""not_set""", default=None)
    seuil_minimum_relatif: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "mode": [],
        "filename": [],
        "seuil_absolu": [],
        "seuil_relatif": [],
        "seuil_minimum_relatif": [],
    }


################################################################


class Ijk_ft_base(Interprete):
    r"""
    not_set
    """

    check_stats: Optional[bool] = Field(
        description=r"""Flag to compute additional (xy)-plane averaged statistics""",
        default=None,
    )
    dt_post: Optional[int] = Field(
        description=r"""Post-processing frequency (for lata output)""", default=None
    )
    dt_post_stats_plans: Optional[int] = Field(
        description=r"""Post-processing frequency for averaged statistical files (txt files containing averaged information on (xy) planes for each z-center) both instantaneous, or cumulated time-integration (see file header for variables list)""",
        default=None,
    )
    dt_post_stats_bulles: Optional[int] = Field(
        description=r"""Post-processing frequency for bubble information (for out files as bubble area, centroid position, etc...)""",
        default=None,
    )
    champs_a_postraiter: Optional[List[str]] = Field(
        description=r"""List of variables to post-process in lata files.""",
        default=None,
    )
    expression_vx_ana: Optional[str] = Field(
        description=r"""Analytical Vx (parser of x,y,z, t) used for post-processing only""",
        default=None,
    )
    expression_vy_ana: Optional[str] = Field(
        description=r"""Analytical Vy (parser of x,y,z, t) used for post-processing only""",
        default=None,
    )
    expression_vz_ana: Optional[str] = Field(
        description=r"""Analytical Vz (parser of x,y,z, t) used for post-processing only""",
        default=None,
    )
    expression_p_ana: Optional[str] = Field(
        description=r"""analytical pressure solution (parser of x,y,z, t) used for post-processing only""",
        default=None,
    )
    expression_dpdx_ana: Optional[str] = Field(
        description=r"""analytical expression dP/dx=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dpdy_ana: Optional[str] = Field(
        description=r"""analytical expression dP/dy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dpdz_ana: Optional[str] = Field(
        description=r"""analytical expression dP/dz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dudx_ana: Optional[str] = Field(
        description=r"""analytical expression dU/dx=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dudy_ana: Optional[str] = Field(
        description=r"""analytical expression dU/dy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dudz_ana: Optional[str] = Field(
        description=r"""analytical expression dU/dz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dvdx_ana: Optional[str] = Field(
        description=r"""analytical expression dV/dx=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dvdy_ana: Optional[str] = Field(
        description=r"""analytical expression dV/dy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dvdz_ana: Optional[str] = Field(
        description=r"""analytical expression dV/dz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dwdx_ana: Optional[str] = Field(
        description=r"""analytical expression dW/dx=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dwdy_ana: Optional[str] = Field(
        description=r"""analytical expression dW/dy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_dwdz_ana: Optional[str] = Field(
        description=r"""analytical expression dW/dz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdxdx_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dx2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdydy_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dy2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdzdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dz2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdxdy_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dxdy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdxdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dxdz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddpdydz_ana: Optional[str] = Field(
        description=r"""analytical expression d2P/dydz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudxdx_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dx2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudydy_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dy2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudzdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dz2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudxdy_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dxdy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudxdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dxdz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddudydz_ana: Optional[str] = Field(
        description=r"""analytical expression d2U/dydz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdxdx_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dx2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdydy_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dy2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdzdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dz2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdxdy_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dxdy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdxdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dxdz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddvdydz_ana: Optional[str] = Field(
        description=r"""analytical expression d2V/dydz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdxdx_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dx2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdydy_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dy2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdzdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dz2=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdxdy_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dxdy=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdxdz_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dxdz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    expression_ddwdydz_ana: Optional[str] = Field(
        description=r"""analytical expression d2W/dydz=f(x,y,z,t), for post-processing only""",
        default=None,
    )
    t_debut_statistiques: Optional[float] = Field(
        description=r"""Initial time for computation, printing and accumulating time-integration""",
        default=None,
    )
    sondes: Optional[Bloc_lecture] = Field(description=r"""probes""", default=None)
    p_seuil_max: Optional[float] = Field(
        description=r"""not_set, default 10000000""", default=None
    )
    p_seuil_min: Optional[float] = Field(
        description=r"""not_set, default -10000000""", default=None
    )
    coef_ammortissement: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    coef_immobilisation: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    coef_mean_force: Optional[float] = Field(description=r"""not_set""", default=None)
    coef_force_time_n: Optional[float] = Field(description=r"""not_set""", default=None)
    coef_rayon_force_rappel: Optional[float] = Field(
        description=r"""not_set""", default=None
    )
    ijk_splitting: Literal["grid_splitting"] = Field(
        description=r"""Definition of domain decomposition for parallel computations""",
        default="grid_splitting",
    )
    ijk_splitting_ft_extension: int = Field(
        description=r"""Number of element used to extend the computational domain at each side of periodic boundary to accommodate for bubble evolution.""",
        default=0,
    )
    tinit: Optional[float] = Field(description=r"""initial time""", default=None)
    timestep: float = Field(description=r"""Upper limit of the timestep""", default=0.0)
    timestep_facsec: Optional[float] = Field(
        description=r"""Security factor on timestep""", default=None
    )
    cfl: Optional[float] = Field(
        description=r"""To provide a value of the limiting CFL number used for setting the timestep""",
        default=None,
    )
    fo: Optional[float] = Field(description=r"""not_set""", default=None)
    oh: Optional[float] = Field(description=r"""not_set""", default=None)
    nb_pas_dt_max: int = Field(
        description=r"""maximum limit for the number of timesteps""", default=0
    )
    max_simu_time: Optional[float] = Field(
        description=r"""maximum limit for the simulation time""", default=None
    )
    tstep_init: Optional[int] = Field(
        description=r"""index first interation for recovery""", default=None
    )
    use_tstep_init: Optional[int] = Field(
        description=r"""use tstep init for constant post-processing step""",
        default=None,
    )
    multigrid_solver: Multigrid_solver = Field(
        description=r"""not_set""", default_factory=lambda: eval("Multigrid_solver()")
    )
    check_divergence: Optional[bool] = Field(
        description=r"""Flag to compute and print the value of div(u) after each pressure-correction""",
        default=None,
    )
    vitesse_entree: Optional[float] = Field(
        description=r"""Velocity to prescribe at inlet""", default=None
    )
    vitesse_upstream: Optional[float] = Field(
        description=r"""Velocity to prescribe at 'nb_diam_upstream_' before bubble 0.""",
        default=None,
    )
    upstream_dir: Optional[int] = Field(
        description=r"""Direction to prescribe the velocity""", default=None
    )
    expression_vitesse_upstream: Optional[str] = Field(
        description=r"""Analytical expression to set the upstream velocity""",
        default=None,
    )
    upstream_stencil: Optional[int] = Field(
        description=r"""Width on which the velocity is set""", default=None
    )
    nb_diam_upstream: Optional[float] = Field(
        description=r"""Number of bubble diameters upstream of bubble 0 to prescribe the velocity.""",
        default=None,
    )
    nb_diam_ortho_shear_perio: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    rho_liquide: float = Field(description=r"""liquid density""", default=0.0)
    mu_liquide: float = Field(description=r"""liquid viscosity""", default=0.0)
    check_stop_file: Optional[str] = Field(
        description=r"""stop file to check (if 1 inside this file, stop computation)""",
        default=None,
    )
    dt_sauvegarde: Optional[int] = Field(
        description=r"""saving frequency (writing files for computation restart)""",
        default=None,
    )
    nom_sauvegarde: Optional[str] = Field(
        description=r"""Definition of filename to save the calculation""", default=None
    )
    sauvegarder_xyz: Optional[bool] = Field(
        description=r"""save in xyz format""", default=None
    )
    nom_reprise: Optional[str] = Field(
        description=r"""Enable restart from filename given""", default=None
    )
    gravite: Optional[List[float]] = Field(
        description=r"""gravity vector [gx, gy, gz]""", default=None
    )
    expression_vx_init: Optional[str] = Field(
        description=r"""initial field for x-velocity component (parser of x,y,z)""",
        default=None,
    )
    expression_vy_init: Optional[str] = Field(
        description=r"""initial field for y-velocity component (parser of x,y,z)""",
        default=None,
    )
    expression_vz_init: Optional[str] = Field(
        description=r"""initial field for z-velocity component (parser of x,y,z)""",
        default=None,
    )
    expression_derivee_force: Optional[str] = Field(
        description=r"""expression of the time-derivative of the X-component of a source-term (see terme_force_ini for the initial value). terme_force_ini : initial value of the X-component of the source term (see expression_derivee_force for time evolution)""",
        default=None,
    )
    compute_force_init: Optional[str] = Field(description=r"""not_set""", default=None)
    terme_force_init: Optional[str] = Field(description=r"""not_set""", default=None)
    correction_force: Optional[str] = Field(description=r"""not_set""", default=None)
    vol_bulle_monodisperse: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    diam_bulle_monodisperse: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    coeff_evol_volume: Optional[str] = Field(description=r"""not_set""", default=None)
    vol_bulles: Optional[str] = Field(description=r"""not_set""", default=None)
    time_scheme: Optional[Literal["euler_explicit", "rk3_ft"]] = Field(
        description=r"""Type of time scheme""", default=None
    )
    expression_variable_source_x: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    expression_variable_source_y: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    expression_variable_source_z: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    facteur_variable_source_init: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    expression_derivee_facteur_variable_source: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    expression_p_init: Optional[str] = Field(
        description=r"""initial pressure field (optional)""", default=None
    )
    expression_potential_phi: Optional[str] = Field(
        description=r"""parser to define phi and make a momentum source Nabla phi.""",
        default=None,
    )
    velocity_convection_op: Optional[str] = Field(
        description=r"""Type of velocity convection scheme""", default=None
    )
    sigma: Optional[float] = Field(description=r"""surface tension""", default=None)
    rho_vapeur: Optional[float] = Field(description=r"""vapour density""", default=None)
    mu_vapeur: Optional[float] = Field(
        description=r"""vapour viscosity""", default=None
    )
    interfaces: Optional[Interfaces] = Field(description=r"""not_set""", default=None)
    forcage: Optional[str] = Field(description=r"""not_set""", default=None)
    corrections_qdm: Optional[str] = Field(description=r"""not_set""", default=None)
    thermique: Optional[Annotated[List[Thermique_bloc], "Thermique"]] = Field(
        description=r"""to add energy equation resolution if needed""", default=None
    )
    energie: Optional[str] = Field(description=r"""not_set""", default=None)
    fichier_post: Optional[str] = Field(
        description=r"""name of the post-processing file (lata file)""", default=None
    )
    fichier_reprise_vitesse: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    timestep_reprise_vitesse: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    boundary_conditions: Bloc_lecture = Field(
        description=r"""BC""", default_factory=lambda: eval("Bloc_lecture()")
    )
    disable_solveur_poisson: Optional[bool] = Field(
        description=r"""Disable pressure poisson solver""", default=None
    )
    resolution_fluctuations: Optional[bool] = Field(
        description=r"""Disable pressure poisson solver""", default=None
    )
    disable_diffusion_qdm: Optional[bool] = Field(
        description=r"""Disable diffusion operator in momentum""", default=None
    )
    disable_source_interf: Optional[bool] = Field(
        description=r"""Disable computation of the interfacial source term""",
        default=None,
    )
    disable_convection_qdm: Optional[bool] = Field(
        description=r"""Disable convection operator in momentum""", default=None
    )
    disable_diphasique: Optional[bool] = Field(
        description=r"""Disable all calculations related to interfaces (phase properties, interfacial force, ... )""",
        default=None,
    )
    frozen_velocity: Optional[str] = Field(description=r"""not_set""", default=None)
    velocity_reset: Optional[str] = Field(description=r"""not_set""", default=None)
    improved_initial_pressure_guess: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    include_pressure_gradient_in_ustar: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    use_inv_rho_for_mass_solver_and_calculer_rho_v: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    use_inv_rho_in_poisson_solver: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    diffusion_alternative: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    suppression_rejetons: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    correction_bilan_qdm: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    refuse_patch_conservation_qdm_rk3_source_interf: Optional[bool] = Field(
        description=r"""experimental Keyword, not for use""", default=None
    )
    test_etapes_et_bilan: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    ajout_init_a_reprise: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    harmonic_nu_in_diff_operator: Optional[bool] = Field(
        description=r"""Disable pressure poisson solver""", default=None
    )
    harmonic_nu_in_calc_with_indicatrice: Optional[bool] = Field(
        description=r"""Disable pressure poisson solver""", default=None
    )
    reprise_vap_velocity_tmoy: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    reprise_liq_velocity_tmoy: Optional[str] = Field(
        description=r"""not_set""", default=None
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "check_stats": [],
        "dt_post": [],
        "dt_post_stats_plans": [],
        "dt_post_stats_bulles": [],
        "champs_a_postraiter": [],
        "expression_vx_ana": [],
        "expression_vy_ana": [],
        "expression_vz_ana": [],
        "expression_p_ana": [],
        "expression_dpdx_ana": [],
        "expression_dpdy_ana": [],
        "expression_dpdz_ana": [],
        "expression_dudx_ana": [],
        "expression_dudy_ana": [],
        "expression_dudz_ana": [],
        "expression_dvdx_ana": [],
        "expression_dvdy_ana": [],
        "expression_dvdz_ana": [],
        "expression_dwdx_ana": [],
        "expression_dwdy_ana": [],
        "expression_dwdz_ana": [],
        "expression_ddpdxdx_ana": [],
        "expression_ddpdydy_ana": [],
        "expression_ddpdzdz_ana": [],
        "expression_ddpdxdy_ana": [],
        "expression_ddpdxdz_ana": [],
        "expression_ddpdydz_ana": [],
        "expression_ddudxdx_ana": [],
        "expression_ddudydy_ana": [],
        "expression_ddudzdz_ana": [],
        "expression_ddudxdy_ana": [],
        "expression_ddudxdz_ana": [],
        "expression_ddudydz_ana": [],
        "expression_ddvdxdx_ana": [],
        "expression_ddvdydy_ana": [],
        "expression_ddvdzdz_ana": [],
        "expression_ddvdxdy_ana": [],
        "expression_ddvdxdz_ana": [],
        "expression_ddvdydz_ana": [],
        "expression_ddwdxdx_ana": [],
        "expression_ddwdydy_ana": [],
        "expression_ddwdzdz_ana": [],
        "expression_ddwdxdy_ana": [],
        "expression_ddwdxdz_ana": [],
        "expression_ddwdydz_ana": [],
        "t_debut_statistiques": [],
        "sondes": [],
        "p_seuil_max": [],
        "p_seuil_min": [],
        "coef_ammortissement": [],
        "coef_immobilisation": [],
        "coef_mean_force": [],
        "coef_force_time_n": [],
        "coef_rayon_force_rappel": [],
        "ijk_splitting": [],
        "ijk_splitting_ft_extension": [],
        "tinit": [],
        "timestep": [],
        "timestep_facsec": [],
        "cfl": [],
        "fo": [],
        "oh": [],
        "nb_pas_dt_max": [],
        "max_simu_time": [],
        "tstep_init": [],
        "use_tstep_init": [],
        "multigrid_solver": [],
        "check_divergence": [],
        "vitesse_entree": [],
        "vitesse_upstream": [],
        "upstream_dir": [],
        "expression_vitesse_upstream": [],
        "upstream_stencil": [],
        "nb_diam_upstream": [],
        "nb_diam_ortho_shear_perio": [],
        "rho_liquide": [],
        "mu_liquide": [],
        "check_stop_file": [],
        "dt_sauvegarde": [],
        "nom_sauvegarde": [],
        "sauvegarder_xyz": [],
        "nom_reprise": [],
        "gravite": [],
        "expression_vx_init": [],
        "expression_vy_init": [],
        "expression_vz_init": [],
        "expression_derivee_force": [],
        "compute_force_init": [],
        "terme_force_init": [],
        "correction_force": [],
        "vol_bulle_monodisperse": [],
        "diam_bulle_monodisperse": [],
        "coeff_evol_volume": [],
        "vol_bulles": [],
        "time_scheme": [],
        "expression_variable_source_x": [],
        "expression_variable_source_y": [],
        "expression_variable_source_z": [],
        "facteur_variable_source_init": [],
        "expression_derivee_facteur_variable_source": [],
        "expression_p_init": [],
        "expression_potential_phi": [],
        "velocity_convection_op": [],
        "sigma": [],
        "rho_vapeur": [],
        "mu_vapeur": [],
        "interfaces": [],
        "forcage": [],
        "corrections_qdm": [],
        "thermique": [],
        "energie": [],
        "fichier_post": [],
        "fichier_reprise_vitesse": [],
        "timestep_reprise_vitesse": [],
        "boundary_conditions": [],
        "disable_solveur_poisson": [],
        "resolution_fluctuations": [],
        "disable_diffusion_qdm": [],
        "disable_source_interf": [],
        "disable_convection_qdm": [],
        "disable_diphasique": [],
        "frozen_velocity": [],
        "velocity_reset": [],
        "improved_initial_pressure_guess": [],
        "include_pressure_gradient_in_ustar": [],
        "use_inv_rho_for_mass_solver_and_calculer_rho_v": [],
        "use_inv_rho_in_poisson_solver": [],
        "diffusion_alternative": [],
        "suppression_rejetons": [],
        "correction_bilan_qdm": [],
        "refuse_patch_conservation_qdm_rk3_source_interf": [],
        "test_etapes_et_bilan": [],
        "ajout_init_a_reprise": [],
        "harmonic_nu_in_diff_operator": [],
        "harmonic_nu_in_calc_with_indicatrice": [],
        "reprise_vap_velocity_tmoy": [],
        "reprise_liq_velocity_tmoy": [],
    }


################################################################


class Echange_contact_vdf_ft_disc_solid(Condlim_base):
    r"""
    echange_conatct_vdf en prescisant la phase
    """

    autre_probleme: str = Field(description=r"""name of other problem""", default="")
    autre_bord: str = Field(description=r"""name of other boundary""", default="")
    autre_champ_temperature_indic1: str = Field(
        description=r"""name of temperature indic 1""", default=""
    )
    autre_champ_temperature_indic0: str = Field(
        description=r"""name of temperature indic 0""", default=""
    )
    autre_champ_indicatrice: str = Field(
        description=r"""name of indicatrice""", default=""
    )
    _synonyms: ClassVar[dict] = {
        None: [],
        "autre_probleme": [],
        "autre_bord": [],
        "autre_champ_temperature_indic1": [],
        "autre_champ_temperature_indic0": [],
        "autre_champ_indicatrice": [],
    }


################################################################


class Ch_front_input_ale(Front_field_base):
    r"""
    Class to define a boundary condition on a moving boundary of a mesh (only for the
    Arbitrary Lagrangian-Eulerian framework ) .

    Example: Ch_front_input_ALE { nb_comp 3 nom VITESSE_IN_ALE probleme pb initial_value 3 1.
    0. 0. }
    """

    _synonyms: ClassVar[dict] = {None: []}


##################################################
### Classes Declaration and Dataset
##################################################
class Declaration(Objet_u):
    """Added Pydantic class to handle forward declaration in the TRUST dataset"""

    ze_type: type = Field(
        description="Class type being read in the forward declaration", default=None
    )
    identifier: str = Field(
        description="Name assigned to the object in the dataset", default="??"
    )


class Read(Interprete):
    """The 'read' instruction in a TRUST dataset. Overriden from the automatic generation to make the second argument a Objet_u.
    See also Read_Parser class in base.py module.
    """

    identifier: str = Field(
        description="Identifier of the class being read. Must match a previous forward Declaration.",
        default="??",
    )
    obj: Objet_u = Field(description="The object being read.", default=None)
    _synonyms: ClassVar[dict] = {None: ["lire"], "identifier": [], "obj": []}


class Dataset(Objet_u):
    """A full TRUST dataset! It is an ordered list of objects."""

    _declarations: Dict[
        str, Any
    ] = {}  # Private - key: declaration name (like 'pb' in 'pb_conduction pb'), value: a couple (cls, index)
    # where 'cls' is a Declaration object, and index is its position in the 'entries' member
    entries: List[Objet_u] = Field(
        description="The objects making up the dataset", default=[]
    )

    def get(self, identifier):
        """User method - Returns the object associated with a name in the data set"""
        from trustify.misc_utilities import TrustifyException

        if identifier not in self._declarations:
            raise TrustifyException(f"Invalid identifer '{identifier}'")
        it_num = self._declarations[identifier][1]
        if it_num < 0:
            raise TrustifyException(
                f"Identifer '{identifier}' has been declared, but has not been read in the dataset (no 'read {identifier} ...' instruction)"
            )
        # Return the object attached into the 'read' instance:
        return self.entries[it_num].obj
