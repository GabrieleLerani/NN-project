import yaml


def load_yaml(path):
    """Function to load the yaml file containing IAAC
    Args:
       path: yaml file path
    Returns:
       Dictionary containing the parameters
    """
    with open(path, "r") as file:
        params_dict = yaml.safe_load(file)
    return params_dict
