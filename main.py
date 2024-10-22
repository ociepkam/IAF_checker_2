import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import webbrowser
import mne
import numpy as np
from yaml import safe_load
from philistine.mne import savgol_iaf
import warnings
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import contextlib
import io


def load_config(proj_name: str) -> Dict:
    """
    Load the configuration from a YAML file for a specific project.

    Args:
        proj_name (str): The name of the project whose configuration is being loaded.

    Returns:
        dict: The configuration dictionary loaded from the YAML file.
    """
    try:
        with open(os.path.join("projects_configs", proj_name)) as yaml_file:
            doc = safe_load(yaml_file)
        return doc
    except FileNotFoundError:
        raise Exception("Can't load config file")


def box_choose_user(user_list: List[str], title: str) -> str:
    """
    Display a GUI dialog to let the user select from a list of users.

    Args:
        user_list (list): List of users to choose from.
        title (str): The title of the selection window.

    Returns:
        str: The selected user name or an empty string if no selection is made.
    """
    selected_value = ""

    def on_ok():
        nonlocal selected_value
        selected_value = combo.get()
        if selected_value:
            root.destroy()
            return selected_value

    def on_exit():
        root.destroy()
        return selected_value

    root = tk.Tk()
    root.title(title)
    root.geometry("300x150")

    # Combobox to display the list of users
    combo = ttk.Combobox(root, values=user_list, state='readonly')
    combo.pack(pady=30)

    # OK and Exit buttons
    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.pack(side=tk.RIGHT, padx=30)

    exit_button = tk.Button(root, text="Exit", command=on_exit)
    exit_button.pack(side=tk.LEFT, padx=30)

    root.mainloop()
    return selected_value


def open_result_file(filename: str) -> pd.DataFrame:
    """
    Open or create a result file as a pandas DataFrame.

    Args:
        filename (str): The path to the result file.

    Returns:
        pd.DataFrame: The DataFrame containing the results.
    """
    if not os.path.isfile(filename):
        # Create a new file with the required columns
        df = pd.DataFrame(columns=["ID", "IAF", "certainty"])
        df.to_csv(filename, index=False)
    else:
        # Load the existing file
        df = pd.read_csv(filename)
    return df


def browse_file(config: Dict) -> Optional[str]:
    """
    Open a file browsing dialog to select an EEG file based on the config.

    Args:
        config (dict): Configuration dictionary containing browsing details.

    Returns:
        str: The selected file path or None if no file is chosen.
    """
    # Create a tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window to show only the dialog

    # Open file dialog in the main thread
    file_name = filedialog.askopenfilename(
        title="Choose a file for verification",
        initialdir=config["browse_directory"],
        filetypes=[("EEG files", "*.edf;*.set;*.csv;*.fif"), ("All files", "*.*")]  # Specify the file types
    )

    # Exit the application if no file was chosen
    if not file_name:
        exit(0)

    root.destroy()  # Destroy the root window after file selection
    return file_name


def files_to_choose(config: Dict, user_name: str) -> List[str]:
    """
    Get a list of EEG files that have not been processed for a specific user.

    Args:
        config (dict): Configuration dictionary.
        user_name (str): The name of the user.

    Returns:
        list: A list of file names to choose from.
    """
    iaf_results = open_result_file(filename=config["results_file"])
    files = os.listdir(config["browse_directory"])
    # Filter files based on file type and processing status
    files = [file for file in files if file.endswith(config["eeg_file_type"])]
    files = [file for file in files if
             file.split(".")[0] not in iaf_results['ID'].tolist() or
             pd.isna(iaf_results.loc[iaf_results['ID'] == file.split(".")[0], user_name].values[0])]
    if not files:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("No Files", "There are no more files to analyze.")
        root.destroy()
        exit(1)
    return files


def choose_file(config: Dict, user_name: str) -> str:
    """
    Choose a file based on the file selection strategy defined in the config.

    Args:
        config (dict): Configuration dictionary with file selection strategy.
        user_name (str): The name of the user.

    Returns:
        str: The path to the selected file.
    """
    if config["file_choice"] == "choice":
        file_name = browse_file(config=config)
    elif config["file_choice"] == "list":
        file_name = files_to_choose(config=config, user_name=user_name)[0]
        file_name = os.path.join(config["browse_directory"], file_name)
    elif config["file_choice"] == "random":
        file_name = np.random.choice(files_to_choose(config=config, user_name=user_name))
        file_name = os.path.join(config["browse_directory"], file_name)
    else:
        raise Exception(f"{config['file_choice']} is an invalid file_choice")
    return file_name


def load_file(file_name: str):
    """
    Load EEG data from a file into an MNE data object.

    Args:
        file_name (str): Path to the EEG file.

    Returns:
        mne.io.Raw or mne.BaseEpochs: MNE data object.
    """
    try:
        data = mne.io.read_raw(file_name, verbose=False)
    except TypeError:
        # Handle specific file types
        if file_name.split(".")[-1] == 'set':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                data = mne.read_epochs_eeglab(file_name, verbose=False)
        else:
            raise ValueError(f"{file_name.split('.')[-1]} with epochs is unsupported.")
    return data


def calculate_iaf_from_list(list_of_iaf: List[Optional[float]]) -> Tuple[Optional[float], float]:
    """
    Calculate the median IAF and certainty from a list of IAF values.

    Args:
        list_of_iaf (list): List of IAF values.

    Returns:
        tuple: The IAF (float) and certainty (float).
    """
    none_ratio = list_of_iaf.count(None) / len(list_of_iaf)
    if none_ratio == 1:
        return None, 0

    filtered_numbers = np.array([num for num in list_of_iaf if num is not None])

    iaf = round(np.median(filtered_numbers) * 4) / 4
    stddev = np.std(filtered_numbers)
    certainty = round(max(0, 1 - none_ratio - stddev / (stddev + 1)), 2)
    return iaf, certainty


def iaf_for_epochs(data, config):
    """
    Calculate IAF for each epoch in the EEG data and return the IAF and certainty.

    Args:
        data (mne.BaseEpochs): EEG epochs data.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (list_of_raw, iaf, certainty).
    """
    list_of_raw = []
    list_of_iaf = []
    freq = data.info["sfreq"]
    for epoch_data in data:
        res = mne.io.RawArray(epoch_data, data.info, verbose=False)
        res = res.crop(config["crop_data"]["beginning"], len(res) / freq - config["crop_data"]["end"], verbose=False)
        # Suppress output from savgol_iaf
        with contextlib.redirect_stdout(io.StringIO()):
            iaf = savgol_iaf(res, fmin=config["iaf_range"][0], fmax=config["iaf_range"][1], ax=False)

        list_of_iaf.append(iaf[0])
        list_of_raw.append(res)

    iaf, certainty = calculate_iaf_from_list(list_of_iaf=list_of_iaf)

    return list_of_raw, iaf, certainty


def calculate_iaf(file_name: str, config: Dict) -> Tuple[Optional[float], Optional[float], np.ndarray, np.ndarray]:
    """
    Calculate IAF, certainty, and PSDs for EEG data from the specified file.

    Args:
        file_name (str): Path to the EEG file.
        config (dict): Configuration dictionary with parameters for IAF calculation.

    Returns:
        tuple: (iaf, certainty, psds, freqs) containing the calculated IAF, certainty, PSDs, and frequencies.
    """
    data = load_file(file_name=file_name)
    print(data.info)
    print(type(data))
    freq = data.info["sfreq"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data.pick_channels(config["channels"], verbose=False)

    if isinstance(data, mne.BaseEpochs):
        list_of_raw, iaf, certainty = iaf_for_epochs(data=data, config=config)
        spectrum = data.compute_psd(method="welch", fmin=config["psd_draw_range"]["min"], fmax=config["psd_draw_range"]["max"])
        data, _ = spectrum.get_data(return_freqs=True)
        psds = np.mean(np.mean(data, axis=0), axis=0)
    elif isinstance(data, mne.io.BaseRaw):
        data = data.crop(config["crop_data"]["beginning"], len(data) / freq - config["crop_data"]["end"])
        with contextlib.redirect_stdout(io.StringIO()):
            iaf = savgol_iaf(data, fmin=config["iaf_range"][0], fmax=config["iaf_range"][1], ax=False)[0]
        certainty = None
        spectrum = data.compute_psd(method="welch", fmin=config["psd_draw_range"]["min"], fmax=config["psd_draw_range"]["max"])
        data, _ = spectrum.get_data(return_freqs=True)
        psds = np.mean(data, axis=0)
    else:
        raise Exception("Unknown data type (not raw and no epochs)! <- should never appear")

    return iaf, certainty, psds, spectrum.freqs


def prepare_person_data(config: Dict, user_name: str) -> Tuple[str, Optional[float], Optional[float], np.ndarray, np.ndarray]:
    """
    Prepare data for a specific participant by loading the EEG file and calculating IAF.

    Args:
        config (dict): Configuration dictionary.
        user_name (str): The name of the user.

    Returns:
        tuple: (participant_id, iaf, certainty, psds, freqs).
    """
    file_name = choose_file(config=config, user_name=user_name)
    participant_id = file_name.split('\\')[-1].split(".")[0]
    iaf, certainty, psds, freqs = calculate_iaf(file_name=file_name, config=config)

    iaf_results = open_result_file(filename=config["results_file"])
    if participant_id not in iaf_results['ID'].values:
        new_row = {'ID': participant_id, 'IAF': iaf, 'certainty': certainty}
        iaf_results = pd.concat([iaf_results, pd.DataFrame([new_row])], ignore_index=True)
        iaf_results.to_csv(config["results_file"], index=False)

    return participant_id, iaf, certainty, psds, freqs


def draw_plot(iaf: Optional[float], certainty: Optional[float], psds: np.ndarray, freqs: np.ndarray, participant_data: str) -> go.Figure:
    """
    Draw a plot of the IAF results.

    Args:
        iaf (float): The IAF value.
        certainty (float): The certainty value.
        psds (np.ndarray): Power Spectral Density values.
        freqs (np.ndarray): Frequency values corresponding to the PSD.
        participant_data (str): Information about the participant.

    Returns:
        go.Figure: The figure object containing the plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(freqs), y=list(psds), mode='lines'))
    fig.update_xaxes(tickmode='linear', dtick=1)
    fig.add_annotation(x=0, y=1.3, xref="paper", yref="paper",
                       text=f"<span style='font-family:monospace;'>"
                            f"Analyzed file: {participant_data} <br>"
                            f"IAF from algorithm: {iaf} <br>"
                            f"Certainty of calculations: {certainty}"
                            f"</span>",
                       align='left',
                       font=dict(size=14),
                       showarrow=False)
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [{'yaxis.type': 'linear'}], 'label': 'Linear Scale', 'method': 'relayout'},
                {'args': [{'yaxis.type': 'log'}], 'label': 'Log Scale', 'method': 'relayout'}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0.5,
            'xanchor': 'center',
            'y': 1.3,
            'yanchor': 'top'
        }]
    )
    return fig


def create_app_layout(config: Dict, user_name: str) -> dash.Dash:
    """
    Create the layout for the Dash app.

    Args:
        config (dict): Configuration dictionary.
        user_name (str): The name of the user.

    Returns:
        dash.Dash: The Dash app instance with the specified layout.
    """
    participant_id, iaf, certainty, psds, freqs = prepare_person_data(config=config, user_name=user_name)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Store(id='config', data=config),
        dcc.Store(id='user-name', data=user_name),
        dcc.Store(id='participant-id', data=participant_id),

        html.Div([
            dcc.Graph(id='graph', figure=draw_plot(iaf, certainty, psds, freqs, participant_id), style={'height': '800px'})
        ], style={'width': '100%', 'margin-bottom': '20px'}),

        html.Div([
            dcc.Input(id='input-text', type='text', value='', placeholder='Enter value'),
            html.Button('Submit', id='submit-button', n_clicks=0),
            html.Button('Next', id='next-button', n_clicks=0),
            html.Div(id='output-text')
        ], style={'width': '100%', 'textAlign': 'center'}),

        html.Div([
            dcc.Input(id='input-text-extra', type='text', value='', placeholder='Extra info'),
        ], style={'width': '100%', 'textAlign': 'center'})

    ])
    return app


def register_callbacks(app: dash.Dash):
    """
    Register the callbacks for the Dash app.

    Args:
        app (dash.Dash): The Dash app instance.
    """

    @app.callback(
        [Output('graph', 'figure'),
         Output('output-text', 'children'),
         Output('input-text', 'value'),
         Output('input-text-extra', 'value'),
         Output('participant-id', 'data')],
        [Input('next-button', 'n_clicks'),
         Input('submit-button', 'n_clicks')],
        [State('input-text', 'value'),
         State('input-text-extra', 'value'),
         State('config', 'data'),
         State('user-name', 'data'),
         State('participant-id', 'data')]
    )
    def update_plot_or_save(_next_clicks: int, submit_clicks: int, value: str, extra_info: str, config: Dict, user_name: str, participant_id: str) \
            -> Tuple[go.Figure, str, str, str, str]:
        """
        Update the plot or save data based on user input.

        Args:
            _next_clicks (int): Number of clicks on the Next button (not used in this function).
            submit_clicks (int): Number of clicks on the Submit button.
            value (str): The current value in the input text field.
            extra_info (str): Extra information added by user
            config (dict): Configuration data.
            user_name (str): The name of the user.
            participant_id (str): The ID of the participant.

        Returns:
            tuple: (updated figure, output text, output extra info, input value, participant id).
        """
        ctx = dash.callback_context

        if ctx.triggered and 'submit-button' in ctx.triggered[0]['prop_id']:
            if submit_clicks > 0 and value:
                iaf_results = open_result_file(config["results_file"])
                iaf_results.loc[iaf_results['ID'] == participant_id, user_name] = value
                iaf_results.loc[iaf_results['ID'] == participant_id, f"{user_name}_info"] = extra_info
                iaf_results.to_csv(config["results_file"], index=False)
                return dash.no_update, f'The value "{value}" has been saved to the file.', value, extra_info, dash.no_update

        if ctx.triggered and 'next-button' in ctx.triggered[0]['prop_id']:
            participant_id, iaf, certainty, psds, freqs = prepare_person_data(config, user_name)
            return draw_plot(iaf, certainty, psds, freqs, participant_id), '', '', '', participant_id  # Clear input text

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


def main():
    """
    Main function to run the application.
    This function handles loading configurations, user selections, and initializing the Dash app.
    """
    proj_name = box_choose_user(user_list=os.listdir("projects_configs"), title="Choose project")

    config = load_config(proj_name=proj_name)
    iaf_results = open_result_file(config["results_file"])

    user_name = box_choose_user(user_list=config["user_list"], title="Choose user")
    if user_name not in iaf_results.columns:
        iaf_results[user_name] = ""
        iaf_results.to_csv(config["results_file"], index=False)

    app = create_app_layout(config=config, user_name=user_name)
    register_callbacks(app=app)

    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
