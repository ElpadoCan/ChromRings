# ChromRings
Python package to quantify chromatin reorganisation in *C. elegans* upon starvation. 

## Installation

Please, follow our [Installation guide](https://github.com/ElpadoCan/ChromRings/blob/main/install.rst)

## Usage

To use this tool you will first need to segment the nuclei of the cells of interest (e.g., intestine cells) using our other software called Cell-ACDC. You can find the source code with links to extensive documentation [here](https://github.com/SchmollerLab/Cell_ACDC?tab=readme-ov-file#resources). 

Note that the data to be analysed must reside in the folder `ChromRings/data`.

Once you have the segmentation of the nuclei with the correct data structure (as requested by Cell-ACDC) the chromatin distribution analysis is fully automatic. 

To run the analysis, you need to modify the JSON file called [ChromRings/chromrings/data_info.json](https://github.com/ElpadoCan/ChromRings/blob/main/chromrings/data_info.json) with the location of the data and the pairs of conditions you want to compare. 

Example:

```json
{
    "1_test_3D_vs_2D": {
        "folder_path": "data/1_test_3D_vs_2D",
        "experiments": ["2D_seg/str", "3D_seg/str", "2D_seg/fed", "3D_seg/fed"],
        "channel": "SDC488",
        "plots": [
            "2D_seg/str;;3D_seg/str",
            "2D_seg/fed;;3D_seg/fed"
        ],
        "figs": [""], 
        "colors": {
            "2D_seg/str": "firebrick",
            "3D_seg/str": "orangered",
            "2D_seg/fed": "darkturquoise",
            "3D_seg/fed": "royalblue"
        }
    }
}
```

In the example above the folder `data/1_test_3D_vs_2D_01-02-2023` contains the subfolders listed at the `"experiments"` entry (e.g., `ChromRings\data\1_test_3D_vs_2D\2D_seg\fed`). These are the experiment folders containing the individual Position folders generated with Cell-ACDC. 

At the entry `"plots"` you can define the pairs of conditions you want to compare. In any case, you will get one heatmap for each condition (one condition, one experiment folder). 

Once you modified the JSON file, you can run the analysis by running the `ChromRings\chromrings\main.py` file (we recommend using VS code to run Python scripts, alternatively you can run them in the terminal with the command `python "path to python file"`. To plot the results run the `ChromRings\chromrings\0_plot_dataset.py` file. 

## Citing

If you use this tool in your publication, please cite our publication [here](https://www.biorxiv.org/content/10.1101/2023.07.22.550032v1) as follows:

> Al-Refaie, N., Padovani, F., Binando, F., Hornung, J., Zhao, Q., Towbin, B. D., Cenik, E. S., Stroustrup, N.,
Schmoller, K. M., Cabianca, D. S., An mTOR/RNA pol I axis shapes chromatin architecture in response to
fasting. bioRxiv. doi:10.1101/2023.07.22.550032 (2023)
