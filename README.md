# IAF Checker

The **IAF Checker** is an interactive Python application built using Dash, designed to analyze EEG data, calculate Individual Alpha Frequency (IAF), and visualize the results. This tool provides a user-friendly web interface for selecting EEG files, performing calculations, and visualizing the power spectral density (PSD) of brainwave frequencies.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load and process EEG data in formats like `.edf`, `.set`, `.csv`, and `.fif`.
- Automatically calculate the Individual Alpha Frequency (IAF) for each participant.
- Visualize the Power Spectral Density (PSD) and IAF using interactive plots.
- Keep track of IAF results in a `.csv` file.
- Allows users to interactively select files for processing and manage participant data.

## Requirements

To run this project, ensure you have the following Python packages installed:

- `dash`
- `pandas`
- `numpy`
- `mne`
- `pyyaml`
- `tkinter`
- `philistine` (for IAF calculation)
- `plotly`


## Installation

1. Clone the repository or download the source code:
   ```bash
   git clone https://github.com/ociepkam/IAF_checker_2.git
   cd <project-directory>
   ```

2. Install the necessary dependencies using pip:
   ```bash
   pip install dash pandas numpy mne pyyaml philistine plotly
   ```

## Project Structure

- **`example_EEG_data/`**: Contains sample EEG files for three participants. These files will be used for IAF analysis.
- **`projects_configs/`**: Contains configuration files for various EEG analysis projects. These files define the settings for analyzing EEG data. The folder includes an example configuration file `example.yaml`.
- **`results/`**: This folder is where analysis results will be saved for each project.
- **`main.py`**: The main Python script that runs the entire IAF analysis tool.
- **`check IAF.bat`**: A Windows batch file to quickly launch the application.
- **`README.md`**: Documentation for the project.
- **`requirements.txt`**: A list of Python dependencies for the project.

## Usage

1. **Run the application**:
   ```bash
   python test_dash_3.py
   ```

2. **Open the application in a browser**: The application will automatically launch in your default browser at `http://127.0.0.1:8050/`.

3. **Select a Project**: The application will prompt you to choose a project configuration file from the `projects_configs` directory.

4. **Select a User**: You will be prompted to select a user for whom IAF calculations will be done.

5. **Analyze EEG Files**: The app will allow you to analyze EEG data files and display a graph of the calculated IAF and PSD. You can submit values and save the results for each participant.

6. **Next/Submit**: Use the **Next** button to switch to the next EEG file for analysis, and the **Submit** button to save results for the current participant.

## Configuration

A configuration file (`config.yaml`) is required for the application to run properly. It should be placed in the `projects_configs` directory. Below is a sample configuration structure:

```yaml
browse_directory: example_EEG_data\
eeg_file_type: set
file_choice: random # Options:
                  #   list: automatically choose files by name from not analyzed
                  #   random: choose random file not analyzed
                  #   choice: browser folder to choose file

results_file: results\example_results.csv
user_list: [User1, User2, User3]

crop_data: {beginning: 1, end: 1} # in seconds
iaf_range: [7, 14] # Frequency range for IAF calculation in Hz
channels: ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8"]

psd_draw_range: {min: 2, max: 30} # Frequency range for drawing the PSD graph
```

- **browse_directory**: The directory where EEG files are located.
- **results_file**: The file where IAF results will be saved.
- **eeg_file_type**: The file format of the EEG data (e.g., `.edf`, `.set`).
- **file_choice**: The strategy for selecting files (`choice`, `list`, `random`).
- **channels**: EEG channels to use for the analysis.
- **crop_data**: Data cropping parameters to exclude the beginning and end portions of the file.
- **iaf_range**: The frequency range to calculate IAF.
- **psd_draw_range**: The frequency range for drawing the PSD graph.
- **user_list**: The list of users for whom analysis can be performed.

## How It Works

1. **File Selection**: The application reads EEG files from the directory specified in the configuration.
2. **IAF Calculation**: It calculates the Individual Alpha Frequency (IAF) for each participant using the `savgol_iaf` function from the `philistine` library.
3. **Data Visualization**: The results, including the IAF and PSD, are visualized using Plotly graphs.
4. **Results Storage**: The results for each participant are saved in a `.csv` file, and the user can interact with the interface to update values.

## Contributing

Contributions are welcome! If you find any bugs or want to enhance the functionality, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
