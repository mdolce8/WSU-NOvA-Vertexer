# plot.py

# Utilities to help plot

import matplotlib.pyplot as plt
import os

class ModeType:
    # Define mode values as class constants
    kUnknownMode = -1
    kQE = 0
    kRes = 1
    kDIS = 2
    kCoh = 3
    kCohElastic = 4
    kElectronScattering = 5
    kIMDAnnihilation = 6
    kInverseBetaDecay = 7
    kGlashowResonance = 8
    kAMNuGamma = 9
    kMEC = 10
    kDiffractive = 11
    kEM = 12
    kWeakMix = 13

    @classmethod
    def get_known_int_modes(cls):
        """Return a list of integer mode values excluding kUnknownMode."""
        # Use list comprehension to gather integer modes
        print('Ignoring UnKnown Mode interactions....')
        int_modes = [value for key, value in vars(cls).items() if isinstance(value, int) and not key.startswith("_")]
        # Remove kUnknownMode from the list
        int_modes.remove(cls.kUnknownMode)
        return int_modes

    # Mapping of mode values to string names
    _mode_names = {
        kUnknownMode: "UnknownMode",
        kQE: "QE",
        kRes: "Res",
        kDIS: "DIS",
        kCoh: "Coh",
        kCohElastic: "CohElastic",
        kElectronScattering: "ElectronScattering",
        kIMDAnnihilation: "IMDAnnihilation",
        kInverseBetaDecay: "InverseBetaDecay",
        kGlashowResonance: "GlashowResonance",
        kAMNuGamma: "AMNuGamma",
        kMEC: "MEC",
        kDiffractive: "Diffractive",
        kEM: "EM",
        kWeakMix: "WeakMix",
    }

    @classmethod
    def name(cls, mode):
        """Return the string name of the mode."""
        return cls._mode_names.get(mode, "UnknownMode")

    @classmethod
    def values(cls):
        """Return a list of all mode values."""
        return list(cls._mode_names.keys())

    @classmethod
    def items(cls):
        """Return a list of tuples (mode_value, mode_name)."""
        return [(value, name) for value, name in cls._mode_names.items()]


class NuModeColors:
    def __init__(self):
        # colors for the Elastic Arms Reco.
        self.mode_colors_EA =  {
            'QE': 'royalblue',
            'MEC': 'gold',
            'DIS': 'silver',
            'CohElastic': 'green',
            'Res': 'limegreen',
            'Coh': 'lightcoral',
            'ElectronScattering': 'purple',
            'IMDAnnihilation': 'pink',
            'InverseBetaDecay': 'chocolate',
            'GlashowResonance': 'cyan',
            'AMNuGamma': 'magenta',
            'Diffractive': 'dimgray',
            'EM': 'khaki',
            'WeakMix': 'teal'
               }

        # colors for the model predictions
        # these selected colors are darker shades from the Elastic Arms one.
        self.mode_colors_Model = {
            'QE': 'navy',
            'MEC': 'darkgoldenrod',
            'DIS': 'grey',
            'CohElastic': 'darkgreen',
            'Res': 'forestgreen',
            'Coh': 'firebrick',
            'ElectronScattering': 'indigo',
            'IMDAnnihilation': 'palevioletred',
            'InverseBetaDecay': 'brown',
            'GlashowResonance': 'cadetblue',
            'AMNuGamma': 'darkmagenta',
            'Diffractive': 'black',
            'EM': 'olive',
            'WeakMix': 'darkslategrey'
                     }

    def get_color(self, mode, model=False):
        """Get the color for a specified Interaction Mode. set model=True for model colors"""
        if model:
            return self.mode_colors_Model.get(mode, 'unknown')
        return self.mode_colors_EA.get(mode, 'unknown')


def plot_training_metrics(history, base_dir, output_name):
    # Extract metrics from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse = history.history['mse']
    val_mse = history.history['val_mse']

    # Create a figure and axis object
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot loss and validation loss on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss, label='Training Loss', color=color)
    ax1.plot(val_loss, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Create a second y-axis for MSE
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Mean Squared Error', color=color)
    ax2.plot(mse, label='Training MSE', color=color)
    ax2.plot(val_mse, label='Validation MSE', color='tab:red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Show the plot with a tight layout
    plt.title('Metrics')
    plt.tight_layout()
    plt.show()

    # Ensure plot directory exists
    plot_dir = os.path.join(base_dir, output_name)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
        print('Created directory for plots:', plot_dir)

    # Save the plot in different formats
    for ext in ['png', 'pdf']:
        filename = plot_dir + "/" + output_name + "." + ext
        plt.savefig(filename)
        print("Saved file: ", filename)
    return None

def make_output_dir(outdir_prefix, outdir_name, filename):
    outdir = (outdir_prefix + '/'
              + filename + '/'
              + outdir_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('created dir: {}'.format(outdir))
    else:
        print('dir already exists: {}'.format(outdir))
    return outdir