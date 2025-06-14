# Packaging the dataset

This packages the CREStE dataset into a zip file for easy distribution. The script `package_data.py` is used to create a mini version of the dataset, which includes a subset of the data for quick testing and development.

```bash
python scripts/release/package_data.py --dataset_root ./data/creste_rlang --N 50 --W 100 --output_zip ./data/creste_mini_release.zip
```