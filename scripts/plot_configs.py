from matplotlib import cm

SET2COLORS = cm.get_cmap('Set2').colors
PAIREDCOLORS = cm.get_cmap('Paired').colors
SET2 = {'darkgreen': SET2COLORS[0],
        'orange': SET2COLORS[1],
        'blue': SET2COLORS[2],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET2COLORS[6],
        'grey': SET2COLORS[7],
        }

corl_cp_arxiv_configs = {
    'ub': ('upper bound', SET2['grey']),
    's': ('state CV', SET2['blue']),

    'sa-mc': ('state-action CV (MC)', 'darkorange'),
    'traj-mc': ('TrajCV (MC)', 'tomato'),

    'sa-diff': ('state-action CV (diff)', SET2['lightgreen']),
    'traj-diff': ('TrajCV (diff)', SET2['darkgreen']),
    
    'sa-diff-gn': ('state-action CV (diff-GN)', 'darkorange'),
    'traj-diff-gn': ('TrajCV (diff-GN)', 'tomato'),

    'sa-next': ('state-action CV (next)', SET2['lightgreen']),
    'traj-next': ('TrajCV (next)', SET2['darkgreen']),
    'sa-next-gn': ('state-action CV (next-GN)', 'darkorange'),
    'traj-next-gn': ('TrajCV (next-GN)', 'tomato'),
    'mc': ('Monte Carlo', SET2['brown']),
    'order': [
        'ub', 'mc', 's',
        'sa-mc', 'traj-mc',
        'sa-diff', 'traj-diff', 'sa-diff-gn',  'traj-diff-gn',
        'sa-next', 'traj-next', 'sa-next-gn', 'traj-next-gn',
    ]
}

corl_sigma_configs = {
    'sigma_s_mc': (r"$\mathbb{V}_{S_t}$", SET2['blue']),
    'sigma_a_mc': (r"$\mathbb{V}_{A_t|S_t}$", SET2['lightgreen']),
    'sigma_tau_mc': (r"$\mathbb{V}_{|A_t, S_t}$", SET2['orange']),
    'order': [
        'sigma_s_mc', 'sigma_a_mc', 'sigma_tau_mc'
    ]
}

corl_cp_configs = {
    'upper_bound': ('upper bound', SET2['grey']),
    'state_cv': ('state CV', SET2['orange']),
    'state-action_cv': ('state-action CV', SET2['darkgreen']),
    'traj_cv': ('TrajCV', SET2['blue']),
    'mc': ('Monte Carlo', SET2['pink']),
    'order': [
        'upper_bound', 'mc', 'state_cv', 'state-action_cv', 'traj_cv',
    ]
}


class Configs(object):
    def __init__(self, style=None, colormap='tab10'):
        if not style:
            self.configs = None
            self.colors = iter(cm.get_cmap(colormap).colors * 100)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)
