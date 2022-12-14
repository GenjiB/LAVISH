import numpy as np
import matplotlib.pyplot as plt

from ipdb import set_trace


results = {
    'Vision ': [2, 2, 2, 2, 2, 7,7,7,9,9],
    'Audio ': [24, 24, 12, 12, 7, 7,7,7,9,9],
    'Vision+Audio ': [24, 19, 19, 12, 7, 7,7,7,15,15],
}


def mapping_label(index, name):
    empty = []
    for idx in index:
        empty.append(name[idx])
    return empty

def visualization_temproal(results, path):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    category_names = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
				  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
				  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
				  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
				  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
				  'Clapping']
    
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    category_colors = plt.get_cmap('hsv')(
        np.linspace(0, 1, len(category_names)))

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=3, fancybox=True, shadow=True)

    for legend_idx in range(25):
        ax.barh(labels, 0, left=0, height=0.5, label=category_names[legend_idx], color=category_colors[legend_idx])
    for idx in range(10):
        # ax.barh(labels, widths, left=starts, height=0.5,label=colname, color=color)
        
        colname = mapping_label(data[:,idx], category_names)
        color = mapping_label(data[:,idx], category_colors)
        
        ax.barh(labels, 13.5, left=idx*13.5, height=0.5, color=np.array(color))
        # xcenters = idx + widths / 2

        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, widths)):
        #     ax.text(x, y, str(int(c)), ha='center', va='center',
        #             color=text_color)
    
    ax.legend(ncol=10, bbox_to_anchor=(0, 1.15),
              loc='upper left', fontsize='small')

    plt.savefig(path)
    # return fig, ax


# survey(results, category_names)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left',
#            ncol=1, mode="expand", borderaxespad=0.)
# 