# ChromRings
Python package used to extract all radial intensity profiles from segmented objects. 

This package was used to quantify chromatin reorganisation in *C. elegans* upon starvation. See paper on [Nature Cell Biology](https://www.nature.com/articles/s41556-024-01512-w)

## Installation

Please, follow our [Installation guide](https://github.com/ElpadoCan/ChromRings/blob/main/install.rst)

## Usage

To use this tool you will first need to segment the nuclei of the cells of interest (e.g., intestine cells) using our other software called Cell-ACDC. You can find the source code with links to extensive documentation [here](https://github.com/SchmollerLab/Cell_ACDC?tab=readme-ov-file#resources). 

Note that the data to be analysed must reside in the folder `ChromRings/data`.

Once you have the segmentation of the nuclei with the correct data structure (as requested by Cell-ACDC) the chromatin distribution analysis is fully automatic. 

To run the analysis, you need to create a JSON file in the following location:

```
ChromRings/chromrings/data_info.json
```

This file contais information about the location of the data and the pairs of conditions you want to compare. You can find example JSON files in this folder 
[ChromRings/examples_data_info_jsons](https://github.com/ElpadoCan/ChromRings/examples_data_info_jsons).

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

The JSON file will contain information about all the experiments you want or you will analyse. 

Next, rename the file `ChromRings\chromrings\_example_current_analysis.py` to `current_analysis.py`. Finally, you need to tell the software which experiment you want to analyse. To do so you need to modify the following variable in the `ChromRings\chromrings\current_analysis.py`:

```
batch_name = '1_test_3D_vs_2D' 
```

where the `batch_name` is the experiment you want to analyse, in this example the `1_test_3D_vs_2D` experiment.

Next, you can run the analysis by running the `ChromRings\chromrings\main.py` file (we recommend using VS code to run Python scripts, alternatively you can run them in the terminal with the command `python "path to python file"`. To plot the results run the `ChromRings\chromrings\plot\0_plot_dataset.py` file. 

## Citing

If you use this tool in your publication, please cite our publication [here](https://doi.org/10.1038/s41556-024-01512-w) as follows:

> Al-Refaie, N., Padovani, F., Hornung, J., Pudelko, L., Binando, F., Fabregat d.C., Zhao, Q., Towbin, B. D., Cenik, E. S., Stroustrup, N., Padeken, J., Schmoller, K. M., Cabianca, D. S., Fasting shapes chromatin architecture through an mTOR/RNA Pol I axis. *Nat Cell Biol* **26**, 1903–1917 (2024).

DOI: [10.1038/s41556-024-01512-w](https://doi.org/10.1038/s41556-024-01512-w)
