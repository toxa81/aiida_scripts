import mod_eos
import numpy as np
import matplotlib.pyplot as plt
import click

def add_to_plot(p, eos_pk, zero, do_fit):
    n = load_node(eos_pk)
    structure = n.get_inputs_dict()['structure']
    code = n.get_inputs_dict()['code']
    partition = n.get_inputs_dict()['partition']

    num_atoms = len(structure.sites)
    print("structure: %s, number of atoms: %i"%(structure.get_formula(), len(structure.sites)))

    x = []
    y = []
    for e in n.get_outputs():
        if isinstance(e, JobCalculation):
            print("calculation PK: %i"%e.pk)
            V = e.res.volume
            E = e.res.energy
            Eha = e.res.energy_hartree
            Eoe = e.res.energy_one_electron
            Esm = e.res.energy_smearing
            Eew = e.res.energy_ewald
            Exc = e.res.energy_xc

            x.append(V)
            y.append(E)
            #y.append(calc.res.energy_hartree)
            #y.append(calc.res.energy_one_electron)
            print("volume: %f, energy: %f"%(V, E))

    x, y = zip(*sorted(zip(x, y)))

    if do_fit:
        fit_params, covariance = mod_eos.fit_birch_murnaghan_params(x, y)
        if zero:
            y = [e - fit_params[0] for e in y]
            fit_params[0] = 0
        print("fit parameters: {}".format(fit_params))
    else:
        if zero:
            ymin = min(y)
            y = [e - ymin for e in y]

    V0 = x[0]
    V1 = x[-1]
    
    print("V0=%f, V1=%f, Vinput=%f %f"%(V0, V1, V1 / 1.06, V0 / 0.94))
    
    new_x = np.arange(V0, V1, 0.01)
    
    label = 'raw ' + code.label + " (EoS PK=%i, %s partition)"%(eos_pk, str(partition))
    p.plot(x, y, "o", linewidth = 2.0, label=label)
    if not do_fit:
        p.plot(x, y, linewidth = 2.0)
        
    #plt.plot(x, y, linewidth = 2.0, label=label)
    
    if do_fit:
        label = 'fit ' + code.label + " (EoS PK=%i, %s partition)"%(eos_pk, str(partition))
        p.plot(new_x, mod_eos.birch_murnaghan(new_x, *fit_params),label=label)

    return structure

@click.command()
#@click.option('--eos_pk', type=int, help='PK of the EOS', required=True, multiple=True)
@click.option('--do_fit', type=bool, default=True)
@click.argument('eos_pk', type=int, required=True, nargs=-1)
def plot_eos(eos_pk, do_fit):
    plt.clf()

    f, axarr = plt.subplots(2, sharex=True)
    #axarr[0].plot(x, y)
    axarr[0].set_title('EoS')
    #axarr[1].scatter(x, y)
    
    for i in (0, 1):
        box = axarr[i].get_position()
        axarr[i].set_position([box.x0 + 0.04, box.y0, box.width * 0.95, box.height])
        axarr[i].grid(which = "major", axis = "both")

    for i in eos_pk:
        struct = add_to_plot(axarr[0], i, False, do_fit)

    for i in eos_pk:
        struct = add_to_plot(axarr[1], i, True, do_fit)

    #plt.legend(loc=2, bbox_to_anchor=(1, 1), prop={'size':8})
    axarr[0].legend(loc=2, prop={'size':8})
    #plt.grid(which = "major", axis = "both")

    #axarr[0].xlabel('Unit cell volume [A^3]')
    #plt.ylabel('Total energy [eV]')
    #plt.title("EOS for %s"%struct.get_formula())
    axarr[0].set_title("EOS for %s (structure PK: %i), absolute"%(struct.get_formula(), struct.pk))
    axarr[1].set_title("EOS for %s (structure PK: %i), relative"%(struct.get_formula(), struct.pk))
    #plt.ylim(ymin=-0.0)
    f.text(0.5, 0.04, 'Unit cell volume [A^3]', ha='center', va='center')
    f.text(0.04, 0.5, 'Total energy [eV]', ha='center', va='center',  rotation=90)
    plt.savefig("eos.pdf", format="pdf")

if __name__ == "__main__":
    plot_eos()

