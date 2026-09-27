# Evolving Federated Density Peak Clustering with Shuffler-Based Anonymization

Code for the eFDPC paper (Evolving Systems, 2026, [doi:10.1007/s12530-026-09884-0](https://doi.org/10.1007/s12530-026-09884-0)).

- `model/` – eFDPC method classes
- `federated_server.py` – main federated code
- `demo_eFDPC.ipynb` – basic demo
- `main_experiments.ipynb` – paper experiments (datasets download in `download_datasets.ipynb`)
- `related_methods/` – baseline related methods (could not include directly due to licences, added download links)

```bash
pip install -r requirements.txt
jupyter notebook demo_eFDPC.ipynb
```

```bibtex
@article{ozbot2026efdpc,
  title   = {Evolving federated density peak clustering with shuffler-based anonymization},
  author  = {O{\v{z}}bot, Miha and Ozawa, Seiichi and {\v{S}}krjanc, Igor},
  journal = {Evolving Systems},
  year    = {2026},
  doi     = {10.1007/s12530-026-09884-0}
}
```
