## 1. Installation

Apart from the standard WHAM installation (see [WHAM installation instructions](https://github.com/WiktorMat/WHAM-Plotter/blob/wham-rebuild/README.md)), you also need to install our Combine workflow:

```bash
mkdir external
cd external
git clone https://github.com/WiktorMat/Combine_HtautauCP
cd ..
```

The workflow can, in principle, be installed anywhere. Its location is specified in the WHAM fitting configuration file, which tells WHAM where to write the datacards. Feel free to choose the directory layout that is most convenient for your setup.

To complete the installation, follow the [installation instructions](https://github.com/WiktorMat/Combine_HtautauCP/blob/main/README.md) provided in **Combine_HtautauCP**. In particular, the package must be installed alongside a standard Combine installation (and therefore within a CMSSW environment).

We recommend creating a symbolic link from your CMSSW CombineHarvester directory:

```bash
ln -s external/Combine_HtautauCP [CMSSW_BASE]/src/CombineHarvester/Combine_HtautauCP
```

Replace `[CMSSW_BASE]` with the path to your CMSSW installation.

## 2. Producing datacards

To produce datacards with WHAM, simply run:

```bash
PYTHONPATH=source python -m wham.cli fit [FIT_CONFIG] --shapes-only
```

Replace `[FIT_CONFIG]` with the path to your fitting configuration file.

If you encounter missing Python modules, make sure you have activated your local Conda or Micromamba environment. There is currently no complete list of Python dependencies, so you may need to install missing packages as they are reported.

As an example, the configuration used for the KinFit studies is:

```text
Configurations/fits/phi_cp_combine_export.yaml
```

You can therefore run:

```bash
PYTHONPATH=source python -m wham.cli fit Configurations/fits/phi_cp_combine_export.yaml --shapes-only
```

Note that the fitting configuration imports the main WHAM analysis configuration. The latter defines the input files, event selections, categories, and all other analysis settings.

For the example above, the main analysis configuration is:

```text
Configurations/tautau_2024_CP.yaml
```

## 3. Running Combine

Now switch to your CMSSW environment:

```bash
cd [CMSSW_BASE]/src
cmsenv
cd [CMSSW_BASE]/src/CombineHarvester/Combine_HtautauCP
ulimit -s unlimited
```

Before running the fit, verify that the required input files are present:

```bash
ls -lh cpdatacards/wham_phi_cp/added_histo_tt-mergeXbins.root
ls -lh configs/harvestDatacards_wham_phi_cp.yml
```

Then start the scan:

```bash
python3 scripts/run_scan.py -c configs/harvestDatacards_wham_phi_cp.yml
```

The output files will be written to the `outputs` directory. For example, the resulting CP scan can be found at:

```text
outputs/wham_phi_cp/alpha_cmb.pdf
```

### Mu-tau CP export

The dedicated non-KinFit mu-tau export uses the CP signal region
(`iso_1 < 0.15`, `mt_1 < 65`) and the current MUFFIN fake-factor model:

```bash
PYTHONPATH=source python -m wham.cli fit \
  Configurations/fits/phi_cp_mutau_combine_export.yaml --shapes-only
```

This stages:

```text
external_examples/Combine_HtautauCP/cpdatacards/wham_phi_cp_mutau/added_histo_mt-mergeXbins.root
```

After copying or linking the staged directory into the CMSSW checkout, run:

```bash
python3 scripts/run_scan.py -c configs/harvestDatacards_wham_phi_cp_mutau.yml
```

The resulting scan is written to:

```text
outputs/wham_phi_cp_mutau/alpha_cmb.pdf
```

> **Warning**
>
> Combine reuses existing output files if they are already present. Therefore, if you rerun the fit with updated datacards, make sure to either remove the previous output directory or change the output path in the configuration file (e.g. `configs/harvestDatacards_wham_phi_cp.yml`). Otherwise, stale results may be reused.
