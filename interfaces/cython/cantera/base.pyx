# This file is part of Cantera. See License.txt in the top-level directory or
# at https://cantera.org/license.txt for license and copyright information.

from collections import defaultdict as _defaultdict
from pathlib import PurePath

cdef class _SolutionBase:
    def __cinit__(self, infile='', name='', adjacent=(), origin=None,
                  source=None, yaml=None, thermo=None, species=(),
                  kinetics=None, reactions=(), **kwargs):

        if 'phaseid' in kwargs:
            if name is not '':
                raise AttributeError('duplicate specification of phase name')

            warnings.warn("Keyword 'name' replaces 'phaseid'",
                          FutureWarning)
            name = kwargs['phaseid']

        if 'phases' in kwargs:
            if len(adjacent)>0:
                raise AttributeError(
                    'duplicate specification of adjacent phases')

            warnings.warn("Keyword 'adjacent' replaces 'phases'",
                          FutureWarning)
            adjacent = kwargs['phases']

        # Shallow copy of an existing Solution (for slicing support)
        cdef _SolutionBase other
        if origin is not None:
            other = <_SolutionBase?>origin

            self.base = other.base
            self.thermo = other.thermo
            self.kinetics = other.kinetics
            self.transport = other.transport
            self._base = other._base
            self._source = other._source
            self._thermo = other._thermo
            self._kinetics = other._kinetics
            self._transport = other._transport

            self.thermo_basis = other.thermo_basis
            self._selected_species = other._selected_species.copy()
            return

        # Assign base and set managers to NULL
        self._base = CxxNewSolution()
        self._source = None
        self.base = self._base.get()
        self.thermo = NULL
        self.kinetics = NULL
        self.transport = NULL

        # Parse inputs
        if isinstance(infile, PurePath):
            infile = str(infile)
        if infile.endswith('.yml') or infile.endswith('.yaml') or yaml:
            self._init_yaml(infile, name, adjacent, yaml)
        elif infile or source:
            self._init_cti_xml(infile, name, adjacent, source)
        elif thermo and species:
            self._init_parts(thermo, species, kinetics, adjacent, reactions)
        else:
            raise ValueError("Arguments are insufficient to define a phase")

        # Initialization of transport is deferred to Transport.__init__
        self.base.setThermo(self._thermo)
        self.base.setKinetics(self._kinetics)

        self._selected_species = np.ndarray(0, dtype=np.uint64)

    def __init__(self, *args, **kwargs):
        if isinstance(self, Transport):
            assert self.transport is not NULL

        name = kwargs.get('name')
        if name is not None:
            self.name = name

    property name:
        """
        The name assigned to this object. The default value corresponds
        to the CTI/XML/YAML input file phase entry.
        """
        def __get__(self):
            return pystr(self.base.name())

        def __set__(self, name):
            self.base.setName(stringify(name))

    property source:
        """
        The source of this object (such as a file name).
        """
        def __get__(self):
            return self._source

    property composite:
        """
        Returns tuple of thermo/kinetics/transport models associated with
        this SolutionBase object.
        """
        def __get__(self):
            thermo = None if self.thermo == NULL \
                else pystr(self.thermo.type())
            kinetics = None if self.kinetics == NULL \
                else pystr(self.kinetics.kineticsType())
            transport = None if self.transport == NULL \
                else pystr(self.transport.transportType())

            return thermo, kinetics, transport

    def _init_yaml(self, infile, name, adjacent, source):
        """
        Instantiate a set of new Cantera C++ objects from a YAML
        phase definition
        """
        cdef CxxAnyMap root
        if infile:
            root = AnyMapFromYamlFile(stringify(infile))
            self._source = infile
        elif source:
            root = AnyMapFromYamlString(stringify(source))
            self._source = 'custom YAML'

        phaseNode = root[stringify("phases")].getMapWhere(stringify("name"),
                                                          stringify(name))

        # Thermo
        if isinstance(self, ThermoPhase):
            self._thermo = newPhase(phaseNode, root)
            self.thermo = self._thermo.get()
        else:
            msg = ("Cannot instantiate a standalone '{}' object; use "
                   "'Solution' instead").format(type(self).__name__)
            raise NotImplementedError(msg)

        # Kinetics
        cdef vector[CxxThermoPhase*] v
        cdef _SolutionBase phase

        if isinstance(self, Kinetics):
            v.push_back(self.thermo)
            for phase in adjacent:
                # adjacent bulk phases for a surface phase
                v.push_back(phase.thermo)
            self._kinetics = newKinetics(v, phaseNode, root)
            self.kinetics = self._kinetics.get()
        else:
            self.kinetics = NULL

    def _init_cti_xml(self, infile, name, adjacent, source):
        """
        Instantiate a set of new Cantera C++ objects from a CTI or XML
        phase definition
        """
        if infile:
            rootNode = CxxGetXmlFile(stringify(infile))
            self._source = infile
        elif source:
            rootNode = CxxGetXmlFromString(stringify(source))
            self._source = 'custom CTI/XML'

        # Get XML data
        cdef XML_Node* phaseNode
        if name:
            phaseNode = rootNode.findID(stringify(name))
        else:
            phaseNode = rootNode.findByName(stringify('phase'))
        if phaseNode is NULL:
            raise ValueError("Couldn't read phase node from XML file")

        # Thermo
        if isinstance(self, ThermoPhase):
            self.thermo = newPhase(deref(phaseNode))
            self._thermo.reset(self.thermo)
        else:
            msg = ("Cannot instantiate a standalone '{}' object; use "
                   "'Solution' instead").format(type(self).__name__)
            raise NotImplementedError(msg)

        # Kinetics
        cdef vector[CxxThermoPhase*] v
        cdef _SolutionBase phase

        if isinstance(self, Kinetics):
            v.push_back(self.thermo)
            for phase in adjacent:
                # adjacent bulk phases for a surface phase
                v.push_back(phase.thermo)
            self.kinetics = newKineticsMgr(deref(phaseNode), v)
            self._kinetics.reset(self.kinetics)
        else:
            self.kinetics = NULL

    def _init_parts(self, thermo, species, kinetics, adjacent, reactions):
        """
        Instantiate a set of new Cantera C++ objects based on a string defining
        the model type and a list of Species objects.
        """
        self._source = 'custom parts'
        self.thermo = newThermoPhase(stringify(thermo))
        self._thermo.reset(self.thermo)
        self.thermo.addUndefinedElements()
        cdef Species S
        for S in species:
            self.thermo.addSpecies(S._species)
        self.thermo.initThermo()

        if not kinetics:
            kinetics = "none"

        cdef ThermoPhase phase
        cdef Reaction reaction
        if isinstance(self, Kinetics):
            self.kinetics = CxxNewKinetics(stringify(kinetics))
            self._kinetics.reset(self.kinetics)
            self.kinetics.addPhase(deref(self.thermo))
            for phase in adjacent:
                # adjacent bulk phases for a surface phase
                self.kinetics.addPhase(deref(phase.thermo))
            self.kinetics.init()
            self.kinetics.skipUndeclaredThirdBodies(True)
            for reaction in reactions:
                self.kinetics.addReaction(reaction._reaction, False)
            self.kinetics.resizeReactions()

    property input_data:
        """
        Get input data corresponding to the current state of this Solution,
        along with any user-specified data provided with its input (YAML)
        definition.
        """
        def __get__(self):
            return anymap_to_dict(self.base.parameters(True))

    def update_user_data(self, data):
        """
        Add the contents of the provided `dict` as additional fields when generating
        YAML phase definition files with `write_yaml` or in the data returned by
        `input_data`. Existing keys with matching names are overwritten.
        """
        self.thermo.input().update(dict_to_anymap(data), False)

    def clear_user_data(self):
        """
        Clear all saved input data, so that the data given by `input_data` or
        `write_yaml` will only include values generated by Cantera based on the
        current object state.
        """
        self.thermo.input().clear()

    def write_yaml(self, filename, phases=None, units=None, precision=None,
                   skip_user_defined=None):
        """
        Write the definition for this phase, any additional phases specified,
        and their species and reactions to the specified file.

        :param filename:
            The name of the output file
        :param phases:
            Additional ThermoPhase / Solution objects to be included in the
            output file
        :param units:
            A `UnitSystem` object or dictionary of the units to be used for
            each dimension. See `YamlWriter.output_units`.
        :param precision:
            For output floating point values, the maximum number of digits to
            the right of the decimal point. The default is 15 digits.
        :param skip_user_defined:
            If `True`, user-defined fields which are not used by Cantera will
            be stripped from the output. These additional contents can also be
            controlled using the `update_user_data` and `clear_user_data` functions.
        """
        Y = YamlWriter()
        Y.add_solution(self)
        if phases is not None:
            if isinstance(phases, _SolutionBase):
                # "phases" is just a single phase object
                Y.add_solution(phases)
            else:
                # Assume that "phases" is an iterable
                for phase in phases:
                    Y.add_solution(phase)
        if units is not None:
            Y.output_units = units
        if precision is not None:
            Y.precision = precision
        if skip_user_defined is not None:
            Y.skip_user_defined = skip_user_defined
        Y.to_file(filename)

    def __getitem__(self, selection):
        copy = self.__class__(origin=self)
        if isinstance(selection, slice):
            selection = range(selection.start or 0,
                              selection.stop or self.n_species,
                              selection.step or 1)
        copy.selected_species = selection
        return copy

    property selected_species:
        def __get__(self):
            return list(self._selected_species)
        def __set__(self, species):
            if isinstance(species, (str, int)):
                species = (species,)
            self._selected_species.resize(len(species))
            for i,spec in enumerate(species):
                self._selected_species[i] = self.species_index(spec)

    def __reduce__(self):
        raise NotImplementedError('Solution object is not picklable')

    def __copy__(self):
        raise NotImplementedError('Solution object is not copyable')
