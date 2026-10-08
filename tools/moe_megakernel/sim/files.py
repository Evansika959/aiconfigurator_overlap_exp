"""Reading and writing the CSV and JSON files the simulator's tools exchange."""

import csv
import json


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_csv(path):
    """Every row of a CSV file, as dicts keyed by its header."""
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields=None):
    """Dict rows to a CSV file. `fields` fixes the columns and their order; by default
    they are the first row's keys."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
