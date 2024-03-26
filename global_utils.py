class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def unpack_dict(d, out={}, prefix=""):
    """Recursively unpack dict"""
    for k, v in d.items():
        if isinstance(v, dict):
            out = unpack_dict(v, out, prefix + k + ".")
        
        else:
            if not k.startswith("__"):
                if isinstance(v, list) or isinstance(v, tuple):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            out = unpack_dict(item, out, prefix + k + f".{i}.")
                        else:
                            out[prefix + k + f".{i}"] = item
                else:
                    out[prefix + k] = v
    return out


def pretty_print_config(config):
    print('cfg:',)
    for k, v in unpack_dict(config, prefix="").items():
        print(k.ljust(33), ':', v)
    print()

