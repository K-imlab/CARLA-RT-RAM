#!/bin/bash

PYTHONPATH=$PYTHONPATH:.:./messages python main.py
PYTHONPATH=$PYTHONPATH:.:./messages fake_data/sender.py
PYTHONPATH=$PYTHONPATH:.:./messages python scripts/make_inputs.py
PYTHONPATH=$PYTHONPATH:.:./messages python scripts/network_forward.py
PYTHONPATH=$PYTHONPATH:.:./messages python scripts/risk_assessment.py
PYTHONPATH=$PYTHONPATH:.:./messages python scripts/visualize.py