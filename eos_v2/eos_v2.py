import click
import numpy
import calculation_helpers
from aiida.work.workchain import WorkChain
from aiida.work.workfunction import workfunction
from aiida.orm.data.base import Float, List, Int, Str
from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.code import Code
from aiida.work.run import submit
from aiida.work.workchain import ToContext

@workfunction
def scaled_structure(structure, scale):

    new_structure = StructureData(cell=numpy.array(structure.cell)*float(scale))

    for site in structure.sites:
        new_structure.append_atom(position=numpy.array(site.position)*float(scale), \
                                  symbols=structure.get_kind(site.kind_name).symbol,\
                                  name=site.kind_name)
    new_structure.label = 'auxiliary structure for EoS'
    new_structure.description = "created from the original structure with PK=%i, "\
                                "lattice constant scaling: %f"%(structure.pk, float(scale))

    return new_structure

def get_pseudos(structure, family_name):
    """
    Set the pseudo to use for all atomic kinds, picking pseudos from the
    family with name family_name.

    :note: The structure must already be set.

    :param family_name: the name of the group containing the pseudos
    """
    from collections import defaultdict
    from aiida.orm.data.upf import get_pseudos_from_structure

    # A dict {kind_name: pseudo_object}
    kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)

    # We have to group the species by pseudo, I use the pseudo PK
    # pseudo_dict will just map PK->pseudo_object
    pseudo_dict = {}
    # Will contain a list of all species of the pseudo with given PK
    pseudo_species = defaultdict(list)

    for kindname, pseudo in kind_pseudo_dict.iteritems():
        pseudo_dict[pseudo.pk] = pseudo
        pseudo_species[pseudo.pk].append(kindname)

    pseudos = {}
    for pseudo_pk in pseudo_dict:
        pseudo = pseudo_dict[pseudo_pk]
        kinds = pseudo_species[pseudo_pk]
        for kind in kinds:
            pseudos[kind] = pseudo

    return pseudos

def get_lapwbasis(structure, family_name):
    lapw_files = {}
    for symb in structure.get_kind_names():
        lapw_files[symb] = None

    LapwbasisData = DataFactory('exciting.lapwbasis')
    group = LapwbasisData.get_lapwbasis_group(family_name)
    
    sp = LapwbasisData.query(dbgroups=group.dbgroup).distinct()
    for s in sp:
        if s.chemical_symbol in lapw_files.keys():
            lapw_files[s.chemical_symbol] = s

    for k,v in lapw_files.iteritems():
        if not v:
            raise RuntimeError("species file for element %s is not found"%k)

    return lapw_files
    
class EoS(WorkChain):
    @classmethod
    def define(cls, spec):
        super(EoS, cls).define(spec)
        spec.outline(cls.submit_jobs, cls.post_process)

    def submit_jobs(self):
        print("structure PK: %i"%self.inputs.structure.pk)

        grp, created = Group.get_or_create(name=self.inputs.group)
        grp.add_nodes([self.calc])
        print("EoS PK: %i"%self.calc.pk)
        
        # get calculation class
        C = CalculationFactory(self.inputs.code.get_input_plugin_name())

        Proc = C.process()

        num_points = 7

        # volume scales from 0.94 to 1.06, alat scales as pow(1/3)
        scales = numpy.linspace(0.94**(1/3.0), 1.06**(1/3.0), num_points).tolist()

        calcs = {}
        
        for scale in scales:
            print("scale = %f"%scale)
            # scaled structure
            new_structure = scaled_structure(self.inputs.structure, Float(scale))
            # basic parameters of the calculation
            params = calculation_helpers.create_calculation_parameters(self.inputs.code,
                                                                       str(self.inputs.partition),
                                                                       int(self.inputs.ranks_per_node),
                                                                       int(self.inputs.ranks_kp),
                                                                       int(self.inputs.ranks_diag))
            
            inputs = Proc.get_inputs_template()
            inputs.code = self.inputs.code

            inputs._options.resources = params['calculation_resources']
            inputs._options.max_wallclock_seconds = 20 * 60
            inputs._options.custom_scheduler_commands = params['custom_scheduler_commands']
            inputs._options.environment_variables = params['environment_variables']
            inputs._options.mpirun_extra_params = params['mpirun_extra_params']

            inputs.structure = new_structure
            inputs.kpoints = KpointsData()
            inputs.kpoints.set_kpoints_mesh(self.inputs.kmesh, offset=(0.0, 0.0, 0.0))

            inputs.parameters = params['calculation_parameters']
            inputs.settings = params['calculation_settings']

            if self.inputs.code.get_input_plugin_name() == 'quantumespresso.pw':
                inputs.pseudo = get_pseudos(new_structure, self.inputs.atomic_files)

            if self.inputs.code.get_input_plugin_name() == 'exciting.exciting':
                inputs.lapwbasis = get_lapwbasis(new_structure, self.inputs.atomic_files)

            future = submit(Proc, **inputs)
            #calcs["s_{}".format(scale)] = future
            
        #return ToContext(**calcs)
    
    def post_process(self):
        return
        #for label in self.ctx:
        #    if "s_" in label:
        #        print "{} {}".format(
        #            label, self.ctx[label]['output_parameters'].dict.energy)

@click.command()
@click.argument('structure_pk', type=int, required=True, nargs=-1)
@click.option('--code', type=str, help='code label', required=True)
@click.option('--atomic_files', type=str, help='label for the atomic files dataset', required=False)
@click.option('--group', type=str, help='label of the EoS workflow group', required=True)
@click.option('--partition', type=str, help='run on "cpu" or "gpu" partition', default='cpu')
@click.option('--ranks_per_node', type=int, help='number of ranks to put on a single node', default=36)
@click.option('--ranks_kp', type=int, help='number of ranks for k-point parallelization', default=1)
@click.option('--ranks_diag', type=int, help='number of ranks for band parallelization', default=36)
@click.option('--kmesh', type=(int, int, int), help='k-point grid', default=(4, 4, 4))
def run(structure_pk, code, atomic_files, group, partition, ranks_per_node, ranks_kp, ranks_diag, kmesh):

    # load code from label@computer
    c = Code.get_from_string(code)
    if not atomic_files:
        if c.get_input_plugin_name() == 'quantumespresso.pw':
            atomic_files = 'SSSP_acc_PBE_fixed'
        if c.get_input_plugin_name() == 'exciting.exciting':
            atomic_files = 'lapw_v1'

    # create k-point mesh
    k = List()
    k.extend(kmesh)
    for spk in structure_pk:
        # load structure from PK
        structure = load_node(spk)
        # create EoS workflow
        eos = EoS()
        eos.run(structure=structure,
                code=c,
                atomic_files=Str(atomic_files),
                group=Str(group),
                kmesh=k,
                partition=Str(partition),
                ranks_per_node=Int(ranks_per_node),
                ranks_kp=Int(ranks_kp),
                ranks_diag=Int(ranks_diag))

if __name__ == "__main__":
    run()

