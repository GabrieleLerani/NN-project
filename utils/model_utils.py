import torch
import torchvision.models as models


def load_model(path):
    """
    Function to save only the model parameters
        Args:
            path: path where to the model
    """
    return torch.load(path)


def save_model(model, path):
    """
    Function to save only the model parameters
        Args:
            model: model parameters as a dictionary
            path: path where to save the entire model
    """
    torch.save(model, path)


def load_model_parameters(model, path):
    """
    Function to save only the model parameters
        Args:
            model: target model
            path: path to the parameters
    """
    model.load_state_dict(torch.load(path))


def save_model_parameters(parameters_dict, path):
    """
    Function to save only the model parameters
        Args:
            parameters_dict: model parameters as a dictionary
            path: path where to save the parameters
    """
    torch.save(parameters_dict, path)
