## Purpose of the project

The project aims to re-implement the architecture of **MinD-Video** described in the [research paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/4e5e0daf4b05d8bfc6377f33fd53a8f4-Paper-Conference.pdf) for reconstructing high-quality videos from brain activity.

### Background

Visual experiences are processed in the brain and can be measured through functional Magnetic Resonance Imaging (fMRI). This data, although complex and high-dimensional, contains rich information about what an individual is seeing or imagining. Translating this brain activity back into understandable visual content poses a significant challenge and has broad applications in neuroscience, cognitive science, and technology development.

### Objectives

1. **Reconstruct Videos from Brain Activity**: Develop a system capable of translating fMRI signals into video sequences, capturing the essence of what the subject is viewing or imagining.
   
2. **Implement Advanced Architectures**: Utilize state-of-the-art machine learning models, including fMRI encoders, diffusion models, and video generation networks, to achieve accurate and high-quality video reconstruction.
   
3. **Understand Brain-Video Mapping**: Explore the underlying mechanisms and correlations between brain activity and visual experiences, contributing to the scientific understanding of perception and cognition.

### Project Structure

The project is organized to support efficient development and collaboration. The structure includes:

- **`docs/`**: Documentation for understanding and using the project.
- **`src/`**: Source code for data preprocessing, model implementation, training, and evaluation.
- **`data/`**: Directories for raw and processed data.
- **`experiments/`**: Separate directories for different experiments and results.
- **`tests/`**: Scripts for testing various components of the system.
- **`notebooks/`**: Jupyter notebooks for exploratory analysis and interactive development.

### Conclusion

The MinD-Video project represents a cutting-edge attempt to bridge the gap between brain activity and visual representation. By re-implementing this architecture, the project seeks to advance the understanding of brain functions and develop practical applications for translating neural data into visual content.
