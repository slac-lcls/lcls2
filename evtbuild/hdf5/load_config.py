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
