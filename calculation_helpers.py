from aiida.orm.data.parameter import ParameterData

def create_calculation_parameters(code, partition, num_ranks_per_node, num_ranks_kp, num_ranks_diag):
    """
    Create a dictionary with parameters for a job execution.
    """

    if partition not in ['cpu', 'gpu']:
        raise RuntimeError('wrong partition name')
    
    num_cores = {'cpu' : 36, 'gpu' : 12}

    if num_cores[partition] % num_ranks_per_node != 0:
        raise RuntimeError('wrong number of ranks per node')
    
    num_threads = num_cores[partition] / num_ranks_per_node

    # total number of ranks
    num_ranks = num_ranks_kp * num_ranks_diag
    
    # get number of nodes
    num_nodes = max(1, num_ranks / num_ranks_per_node)
    
    print("partition: %s"%partition)
    print("number of nodes : %i"%num_nodes)

    # create dictionary to store parameters
    params = {} 

    environment_variables = {'OMP_NUM_THREADS': str(num_threads),\
                             'MKL_NUM_THREADS': str(num_threads),\
                             'KMP_AFFINITY': 'granularity=fine,compact,1'}
    if partition == 'gpu' and num_ranks_per_node > 1:
        environment_variables['CRAY_CUDA_MPS'] = '1'

    params['environment_variables'] = environment_variables

    #calc.set_custom_scheduler_commands('#SBATCH -A, --account=mr21')
    if partition == 'cpu':
        params['custom_scheduler_commands'] = '#SBATCH -C mc'
    if partition == 'gpu':
        params['custom_scheduler_commands'] = '#SBATCH -C gpu'

    # create settings
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        settings = ParameterData(dict={
            'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag)]})
        parameters = ParameterData(dict={
            'CONTROL': {
                'calculation'  : 'scf',
                'restart_mode' : 'from_scratch',
                'disk_io'      : 'none'
                },
            'SYSTEM': {
                'ecutwfc': 40.,
                'ecutrho': 300.,
                #'nbnd': 800,
                'occupations': 'smearing',
                'smearing': 'gauss',
                'degauss': 0.05
                },
            'ELECTRONS': {
                'conv_thr': 1.e-6,
                'electron_maxstep': 100,
                'mixing_beta': 0.7
                }})

    params['calculation_settings'] = settings
    params['calculation_parameters'] = parameters
    params['mpirun_extra_params'] = ['-n', str(num_ranks), '-c', str(num_threads), '--hint=nomultithread','--unbuffered']
    params['calculation_resources'] = {'num_machines': num_nodes, 'num_mpiprocs_per_machine': num_ranks_per_node}
    params['code'] = code

    return params

def create_calculation(structure, params):
    """
    Create calculation object from structure and a dictionary of parameters.
    Calculation has to be stored in DB by the caller.
    """
    code = params['code']
    
    calc = code.new_calc()
    calc.set_max_wallclock_seconds(params.get('calculation_wallclock_seconds', 3600)) # in second
    calc.set_resources(params['calculation_resources'])
    
    calc.use_structure(structure)
    calc.use_parameters(params['calculation_parameters'])
    calc.use_kpoints(params['kpoints'])
    calc.use_settings(params['calculation_settings'])
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        calc.use_pseudos_from_family(params['atomic_files'])
    calc.set_environment_variables(params['environment_variables'])
    calc.set_mpirun_extra_params(params['mpirun_extra_params'])
    calc.set_custom_scheduler_commands(params['custom_scheduler_commands'])

    return calc

