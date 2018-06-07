def load_config(fname):
    config_dict = {}
    with open(fname, 'r') as f:
         for line in f:
             if line[0] in ('#', '\n'):
                continue
             (key, val) = line.split()[:2]
             try:
                     val = eval(val)
             except SyntaxError:
                     pass

             config_dict[key] = val
    return config_dict



def save_config(config, fname):
    with open(fname, 'w') as f:
        for key, value in config.items():
            f.write("%s\t%s\n" % (key, value))

