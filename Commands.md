# Neural Network Project Setup in VS Code

## Overview

This document provides instructions for setting up a neural network (NN) project in Python using Visual Studio Code (VS Code). It covers the installation of necessary tools, setting up a Python virtual environment, and configuring VS Code for development.

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/) installed
- [Python](https://www.python.org/downloads/) installed

## 1. Install VS Code

1. Download and install Visual Studio Code from the [official website](https://code.visualstudio.com/).

## 2. Install Python

1. Ensure Python is installed on your system. Download it from the [official Python website](https://www.python.org/downloads/).

## 3. Install the Python Extension for VS Code

1. Open VS Code.
2. Go to the Extensions view by clicking the Extensions icon in the Activity Bar or by pressing `Ctrl+Shift+X`.
3. Search for "Python" and install the extension provided by Microsoft.

## 4. Set Up a Virtual Environment

1. Open a terminal in VS Code (`Ctrl+`` or through the Terminal menu).
2. Navigate to your project directory or create a new one:
    mkdir nn_project
    cd nn_project
3. Create venv:
    python3 -m venv venv
4. Activate venv:
    source venv/bin/activate
5. Install libraries:
    pip install torch torchvision
6. Save requirements:
    pip freeze > requirements.txt

## 5. Install dependencies from requirements
    pip install -r requirements.txt

## 6. Clone repository
    git clone https://github.com/GabrieleLerani/NN-project
    
## 6. Install extensions on VSCode
    Navigate to the extensions tab and install GitLens Jupyter Python extensions

## 7. Set up GitLens
    Navigate to the Source Control tab and create empty commit
    Then commit and after that sync project






