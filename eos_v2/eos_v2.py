import click
import numpy
import calculation_helpers
from aiida.work.workchain import WorkChain
from aiida.work.workfunction import workfunction
from aiida.orm.data.base import Float, List, Int
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
    new_structure.label = 'created inside EOS run'
    new_structure.description = "auxiliary structure for EOS "\
                                "created from the original structure with PK=%i, "\
                                "lattice constant scaling: %f"%(structure.pk, float(scale))

    return new_structure

#def rescale_structure(structure, scale):
#    the_ase = structure.get_ase()
#    new_ase = the_ase.copy()
#    new_ase.set_cell(the_ase.get_cell() * float(scale), scale_atoms=True)
#    new_structure = DataFactory('structure')(ase=new_ase)
#    return new_structure

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

    
class EoS(WorkChain):
    @classmethod
    def define(cls, spec):
        super(EoS, cls).define(spec)
        spec.outline(cls.submit_jobs, cls.post_process)

    def submit_jobs(self):
        print("structure PK: %i"%self.inputs.structure.pk)
        
        # get calculation class
        C = CalculationFactory(self.inputs.code.get_input_plugin_name())

        Proc = C.process()

        # volume scales from 0.94 to 1.06, alat scales as pow(1/3)
        scales = numpy.linspace(0.979586108715562, 1.019612822422216, 1).tolist()

        calcs = {}
        
        for scale in scales:
            # scaled structure
            new_structure = scaled_structure(self.inputs.structure, Float(scale))
            # basic parameters of the calculation
            params = calculation_helpers.create_calculation_parameters(self.inputs.code, 'cpu', 36, 1, 36)
            #params['kpoints'].set_kpoints_mesh(kwargs.get('kmesh', [1, 1, 1]), offset=(0.0, 0.0, 0.0))
            #params['kpoints'].set_kpoints_mesh([1,1,1], offset=(0.0, 0.0, 0.0))
            #params['atomic_files'] = kwargs['atomic_files']
            #params['atomic_files'] = 'GBRV_lda'
            #params['calculation_wallclock_seconds'] = kwargs.get('time_limit', 3600)
            #params['calculation_wallclock_seconds'] = 3600
            #params['group'] = kwargs['group']
            #params['kpoints'].store()
            #params['calculation_parameters'].store()
            #params['calculation_settings'].store()
            #calculation_helpers.create_calculation(new_structure, params, 'label1', 'desc1')
            
            inputs = Proc.get_inputs_template()
            inputs.code = self.inputs.code

            inputs._options.resources = params['calculation_resources']
            inputs._options.max_wallclock_seconds = 2 * 60
            inputs._options.custom_scheduler_commands = params['custom_scheduler_commands']
            inputs._options.environment_variables = params['environment_variables']
            inputs._options.mpirun_extra_params = params['mpirun_extra_params']

            inputs.structure = new_structure
            inputs.kpoints = KpointsData()
            inputs.kpoints.set_kpoints_mesh([1, 1, 1], offset=(0.0, 0.0, 0.0))

            inputs.parameters = params['calculation_parameters']
            inputs.settings = params['calculation_settings']
            inputs.pseudo = get_pseudos(new_structure, 'GBRV_lda')

            future = submit(Proc, **inputs)
            calcs["s_{}".format(scale)] = future
            
        return ToContext(**calcs)
    
    def post_process(self):
        return
        #for label in self.ctx:
        #    if "s_" in label:
        #        print "{} {}".format(
        #            label, self.ctx[label]['output_parameters'].dict.energy)

@click.command()
@click.option('--structure_pk', type=int, help='PK of the structure', required=True)
##== @click.option('--atomic_files', type=str, help='label for the atomic files dataset', required=True)
##== @click.option('--group', type=str, help='label of the EOS workflow group', required=True)
##== @click.option('--partition', type=str, help='run on "cpu" or "gpu" partition', default='cpu')
##== @click.option('--ranks_per_node', type=int, help='number of ranks to put on a single node', default=36)
##== @click.option('--ranks_kp', type=int, help='number of ranks for k-point parallelization', default=1)
##== @click.option('--ranks_diag', type=int, help='number of ranks for band parallelization', default=36)
##== @click.option('--kmesh', type=(int, int, int), help='k-point grid', default=(4, 4, 4))
#def run(structure_pk, atomic_files, group, partition, ranks_per_node, ranks_kp, ranks_diag, kmesh):
def run(structure_pk):
    #calculation_helpers.submit_eos(structure_pk=structure_pk,
    #                               atomic_files=atomic_files,
    #                               partition=partition,
    #                               num_ranks_per_node=ranks_per_node,
    #                               num_ranks_kp=ranks_kp,
    #                               num_ranks_diag=ranks_diag,
    #                               kmesh=kmesh,
    #                               group=group)

    # load structure from PK
    structure = load_node(structure_pk)

    # load code from label@computer
    code = Code.get_from_string('pw.sirius.x@piz_daint')
    #print(CalculationFactory(code.get_input_plugin_name()))

    eos = EoS()
    eos.run(structure=structure, code=code)
    #print(eos.calc.pk)

if __name__ == "__main__":
    run()
