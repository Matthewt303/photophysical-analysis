import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import os

def collate_paths(folder: str) -> list[str]:

    paths = os.listdir(folder)

    data_paths = [
        os.path.join(folder, path) for path in paths
        if path.endswith(".csv")
    ]

    return sorted(data_paths)

def load_photophys_data(path: str) -> "np.ndarray":

    data = pd.read_csv(path, sep=',', header=None,
                       engine='pyarrow', skiprows=1)
    
    all_data = np.array(data).astype(np.float32)
    return all_data[:, 1:].reshape(-1, 5)

def save_all_results(data_A, data_B, data_C,
                     data_A_files, data_B_files, data_C_files,
                     cond_A, cond_B, cond_C, out):
    
    designate_nt, designate_ct, designate_control = (
        [cond_A] * len(data_A_files)
        [cond_B] * len(data_B_files)
        [cond_C] * len(data_C_files)
    )

    designations = designate_nt + designate_ct + designate_control

    all_results = np.vstack((data_A, data_B, data_C))

    cols = [
        "Duty cycle",
        "Photoswitching time (s)",
        "Number of cycles",
        "Photons per cycle",
        "Total photons"
    ]

    dataframe = pd.DataFrame(all_results, columns=cols)

    dataframe.insert(0, "Designation", designations)

    dataframe.to_csv(os.path.join(out, "all_results.csv"), index=False)

    return dataframe

def save_weighted_stats(means_a, means_b, means_c, stat: str, out:str):

    all_stats = np.vstack((means_a, means_b, means_c))

    with open(os.path.join(out, "weighted_" + stat + ".txt"), "w") as f:

        f.write(str(all_stats))

def plot_all(all_data: "pd.DataFrame", index: int, output_folder: str) -> None:
    """
    Summary:

    Plots a dotplot of the mean FRC resolutions from the noisy data and denoised data.
    The dotplot is saved in a user-specified output folder.
    --------------------------------
    Inputs:

    all_data - a pandas Dataframe that contains three columns. The first column
    specifies the file name, the second specifies the condition, and the third
    specifies the FRC resolution in nanometers

    output_folder - where the plot will be saved. User-specified.

    --------------------------------
    Output:
    None - but two images are saved. One .png and one .svg.

    """

    plt.ioff()

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 28

    sns.set_style("ticks")

    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True

    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_data = all_data[all_data.columns[index]]
    y_label = all_data.columns[index]

    graph = sns.stripplot(
        x=all_data.columns[0],
        y=y_data,
        data=all_data,
        s=15,
        color="midnightblue",
    )

    sns.pointplot(
        data=all_data,
        x=all_data.columns[0],
        y=y_data,
        errorbar="sd",
        markers="_",
        linestyles="none",
        capsize=0.2,
        linewidth=4.0,
        color="darkgreen",
    )

    graph.tick_params(labelsize=28, pad=4)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis="y", which="major", length=6, direction="in")
    ax.tick_params(axis="y", which="minor", length=3, direction="in")
    ax.tick_params(axis="x", which="major", length=6, direction="in")
    ax.tick_params(axis="x", which="minor", length=3, direction="in")

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    #ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2.0)
    ax.spines["top"].set_linewidth(2.0)
    ax.spines["right"].set_linewidth(2.0)
    ax.spines["left"].set_linewidth(2.0)

    ax.set_xlabel("Data type", labelpad=6, fontsize=38)
    ax.set_ylabel(y_label, labelpad=2, fontsize=38)

    out_string = "all_" + all_data.columns[index] + "_plot"

    plt.savefig(os.path.join(output_folder, out_string + ".png"))
    plt.savefig(os.path.join(output_folder, out_string + ".svg"))

def main():

    ### INITIALIZE ###

    nt_data_folder = ""
    ct_data_folder = ""
    control_data_folder = ""
    out_folder = ""

    cond_A = "Ntcys"
    cond_B = "Ctcys"
    cond_C = "Control"
    n_conds = 3

    ### START ###

    nt_paths = collate_paths(nt_data_folder)
    ct_paths = collate_paths(ct_data_folder)
    control_paths = collate_paths(control_data_folder)

    nt_means = np.zeros((len(nt_paths), 5))
    ct_means = np.zeros((len(ct_paths), 5))
    control_means = np.zeros((len(control_paths), 5))

    nt_weights = np.zeros((len(nt_paths), 1))
    ct_weights = np.zeros((len(ct_paths), 1))
    control_weights = np.zeros((len(control_paths), 1))

    for i, files in enumerate(zip(nt_paths, ct_paths, control_paths)):

        nt_data = load_photophys_data(files[0])
        ct_data = load_photophys_data(files[1])
        control_data = load_photophys_data(files[2])

        nt_weights[i, 0] = nt_data.shape[0]
        ct_weights[i, 0] = ct_data.shape[0]
        control_weights[i, 0] = control_data.shape[0]

        nt_means[i, :] = np.mean(nt_data, axis=0)
        ct_means[i, :]  = np.mean(ct_data, axis=0)
        control_means[i, :] = np.mean(control_data, axis=0)
    
    all_data = save_all_results(nt_means, ct_means, control_means,
                                nt_paths, ct_paths, control_paths,
                                cond_A, cond_B, cond_C, out_folder)
    
    nt_weighted_means_num = nt_weights * nt_means
    ct_weighted_means_num = ct_weights * ct_means
    control_weighted_means_num = control_weights * control_means

    weighted_means_nt = np.zeros((1, 5))
    weighted_means_ct = np.zeros((1, 5))
    weighted_means_control = np.zeros((1, 5))

    weighted_means_nt[0, :] = np.sum(nt_weighted_means_num, axis=0) / np.sum(nt_weights)
    weighted_means_ct[0, :] = np.sum(ct_weighted_means_num, axis=0) / np.sum(ct_weights)
    weighted_means_control[0, :] = np.sum(control_weighted_means_num, axis=0)/ np.sum(control_weights)

    save_weighted_stats(weighted_means_nt, weighted_means_ct, weighted_means_control,
                        stat="means", out=out_folder)

    nt_weighted_sd_num = nt_weights * np.square(nt_means - weighted_means_nt)
    ct_weighted_sd_num = ct_weights * np.square(ct_means - weighted_means_ct)
    control_weighted_sd_num = control_weights * np.square(control_means - weighted_means_control)

    weighted_sds_nt = np.zeros((1, 5))
    weighted_sds_ct = np.zeros((1, 5))
    weighted_sds_control = np.zeros((1, 5))

    weighted_sds_nt[0, :] = np.sqrt(np.sum(nt_weighted_sd_num, axis=0) / np.sum(nt_weights))
    weighted_sds_ct[0, :] = np.sqrt(np.sum(ct_weighted_sd_num, axis=0) / np.sum(ct_weights))
    weighted_sds_control[0, :] = np.sqrt(np.sum(control_weighted_sd_num, axis=0) / np.sum(control_weights))

    save_weighted_stats(weighted_means_nt, weighted_means_ct, weighted_means_control,
                        stat="sds", out=out_folder)
    
    for i in range(5):

        plot_all(all_data, i + 1, out_folder)

if __name__ == "__main__":

    main()







